//! The Munich chain ladder (Quarg and Mack 2004): paid and incurred
//! developed together, each age-to-age factor corrected by how far the
//! origin's paid-to-incurred ratio sits from its average at that age
//! (`docs/design/reserving-v02.md`, decision 30).

use prospicio_core::{Lag, Period};

use crate::development::{Average, Development, SigmaInterpolation, interpolate_sigma};
use crate::error::{Error, Result};
use crate::mack::{Mack, MackFit};
use crate::segments::{FitTable, SegmentFits, collect_labelled};
use crate::tail::Tail;
use crate::triangle::{Segment, Triangle};

/// The Munich chain ladder of Quarg and Mack (2004), as R ChainLadder's
/// `MunichChainLadder` and chainladder-python's `MunichAdjustment` build it.
///
/// Paid `P` and incurred `I` are two columns of one triangle, observed at
/// the same cells. Each column has Mack's model (factors `f`, sigmas
/// `sigma`), and at every age `s` the ratios have their own: `q_s = sum P /
/// sum I` and `q^-1_s = sum I / sum P` over the origins observed there,
/// with deviations `rho_I` (of `P/I` around `q`, weighted by `I`) and
/// `rho_P` (of `I/P` around `q^-1`, weighted by `P`). An unestimable `rho`
/// (an age with one origin) is extrapolated log-linearly, as R does.
///
/// The residuals of the link ratios, `(F - f) / sigma * sqrt(C)`, are
/// regressed through the origin on the residuals of the ratios at the
/// link's starting cell, `(I/P - 1/q) / rho_P * sqrt(P)` for paid and
/// `(P/I - q) / rho_I * sqrt(I)` for incurred, over the links whose sigma
/// was estimated (two or more link ratios). The slopes are `lambda_P` and
/// `lambda_I`; each regression leaves out only its own residuals that are
/// not finite (0/0 where a column's sigma is 0), as R's `lm` does. The projection then develops both columns together, step by
/// step from each origin's latest cells:
///
/// `P[s+1] = P[s] (f_P + lambda_P sigma_P / rho_P (I[s]/P[s] - q^-1_s))`
///
/// and the same for incurred with `P/I` and `q`. Under these weights
/// `q^-1` is `1/q` exactly (decision 30, item 3). A tail is one more step,
/// with the tail's factor and sigma and the oldest age's ratios; without a
/// tail the last step is 1 and uncorrected.
///
/// With `lambda` 0 for both, the projection is the two separate chain
/// ladders. When the ratios carry no information (every `rho` zero, as
/// with incurred a fixed multiple of paid), `lambda` is 0.
///
/// Mack's model here is the volume-weighted one (`alpha = 1`), which the
/// residuals assume: another average, a selected pattern, `n_periods` or
/// the drops are an error (decision 30, item 6).
///
/// ```
/// use prospicio_reserving::{DevelopmentColumn, Grain, Long, Month, MunichChainLadder, Triangle};
///
/// let origin = [2020, 2020, 2020, 2020, 2021, 2021, 2021, 2022, 2022, 2023].map(Month::january);
/// let tri = Triangle::from_long(&Long {
///     keys: &[],
///     origin: &origin,
///     development: DevelopmentColumn::Age(&[12, 24, 36, 48, 12, 24, 36, 12, 24, 12]),
///     values: &[
///         ("paid", &[100.0, 150.0, 165.0, 170.0, 110.0, 170.0, 180.0, 120.0, 175.0, 130.0]),
///         ("incurred", &[160.0, 175.0, 172.0, 171.0, 150.0, 190.0, 185.0, 200.0, 185.0, 210.0]),
///     ],
///     origin_grain: Grain::Year,
///     development_grain: Grain::Year,
///     cumulative: true,
/// })?;
/// let fit = MunichChainLadder::default().fit(&tri, "paid", "incurred")?;
/// assert_eq!(fit.ultimate_paid[0], 170.0);
/// assert!(fit.lambda_paid.is_finite() && fit.lambda_incurred.is_finite());
/// # Ok::<(), prospicio_reserving::Error>(())
/// ```
#[derive(Debug, Clone, PartialEq, Default)]
pub struct MunichChainLadder {
    /// Mack's model of the paid column: its sigma interpolation (R's
    /// `est.sigmaP`). The average must be volume-weighted, with no
    /// selected pattern and no exclusions.
    pub paid: Development,
    /// Mack's model of the incurred column (R's `est.sigmaI`), under the
    /// same restrictions.
    pub incurred: Development,
    /// The paid tail (R's `tailP`); the default is none.
    pub tail_paid: Tail,
    /// The incurred tail (R's `tailI`); the default is none.
    pub tail_incurred: Tail,
}

/// A fitted Munich chain ladder of one segment.
#[derive(Debug, Clone, PartialEq)]
pub struct MunichFit {
    /// Origin periods, oldest first; every per-origin vector follows them.
    pub origins: Vec<Period>,
    /// Development ages; every per-age vector follows them.
    pub development: Vec<Lag>,
    /// Mack's model of paid alone: its factors, sigmas, tail and separate
    /// chain-ladder ultimates.
    pub paid: MackFit,
    /// Mack's model of incurred alone.
    pub incurred: MackFit,
    /// `q_s`, the paid-to-incurred ratio at each age: `sum P / sum I` over
    /// the origins observed there (R's `q.f`).
    pub q: Vec<f64>,
    /// `q^-1_s`, the incurred-to-paid ratio at each age, fitted on its own:
    /// `sum I / sum P` (R's `qinverse.f`), which is `1 / q`.
    pub q_inverse: Vec<f64>,
    /// `rho_P`, the deviation of `I/P` around `q^-1` (R's `rhoP.sigma`),
    /// extrapolated log-linearly where it cannot be estimated.
    pub rho_paid: Vec<f64>,
    /// `rho_I`, the deviation of `P/I` around `q` (R's `rhoI.sigma`).
    pub rho_incurred: Vec<f64>,
    /// `lambda_P`, the slope of the paid link-ratio residuals on the
    /// incurred-to-paid residuals; 0 when the ratios carry no information.
    pub lambda_paid: f64,
    /// `lambda_I`, the slope of the incurred link-ratio residuals on the
    /// paid-to-incurred residuals.
    pub lambda_incurred: f64,
    /// The number of residual pairs behind `lambda_paid`.
    pub n_residuals_paid: usize,
    /// The number of residual pairs behind `lambda_incurred`. It can
    /// differ from paid's: a link whose sigma is 0 in one column (every
    /// link ratio the same) has residuals 0/0 there, which leave that
    /// column's regression only, as R's `lm` drops them.
    pub n_residuals_incurred: usize,
    /// Development position of each origin's latest cells.
    pub latest_position: Vec<usize>,
    /// Latest paid per origin.
    pub latest_paid: Vec<f64>,
    /// Latest incurred per origin.
    pub latest_incurred: Vec<f64>,
    /// Munich paid ultimate per origin.
    pub ultimate_paid: Vec<f64>,
    /// Munich incurred ultimate per origin.
    pub ultimate_incurred: Vec<f64>,
}

impl MunichChainLadder {
    /// Fits the `paid` and `incurred` columns of a single-segment triangle.
    ///
    /// Errors when the settings are not Mack's volume-weighted model
    /// (above), when the two columns are not observed at the same cells or
    /// a cell is not positive (the ratios and square roots need both), and
    /// as [`Mack::fit`] on either column.
    pub fn fit(&self, triangle: &Triangle, paid: &str, incurred: &str) -> Result<MunichFit> {
        let p = triangle.segment(paid)?;
        let i = triangle.segment(incurred)?;
        self.fit_pair(&p, &i, paid, incurred)
    }

    /// Fits the two columns in every segment of `triangle`, each on its
    /// own. A failure names its segment.
    pub fn fit_segments(
        &self,
        triangle: &Triangle,
        paid: &str,
        incurred: &str,
    ) -> Result<SegmentFits<MunichFit>> {
        let ps = triangle.segments(paid)?;
        let is = triangle.segments(incurred)?;
        collect_labelled(
            triangle.key_names(),
            triangle.index(),
            ps.iter()
                .zip(&is)
                .map(|(p, i)| self.fit_pair(p, i, paid, incurred)),
        )
    }

    fn fit_pair(&self, p: &Segment, i: &Segment, paid: &str, incurred: &str) -> Result<MunichFit> {
        for dev in [&self.paid, &self.incurred] {
            check_development(dev)?;
        }
        aligned(p, i, paid, incurred)?;
        let mack = |development: &Development, tail: Tail, s: &Segment| {
            Mack {
                development: development.clone(),
                tail,
                ..Default::default()
            }
            .fit_segment(s, &s.ages)
        };
        let paid_fit = mack(&self.paid, self.tail_paid, p)?;
        let incurred_fit = mack(&self.incurred, self.tail_incurred, i)?;

        let n_dev = p.n_dev;
        let n_origins = p.n_origins;
        // The ratios at each age, and their deviations.
        let mut q = Vec::with_capacity(n_dev);
        let mut q_inverse = Vec::with_capacity(n_dev);
        let mut rho_p = Vec::with_capacity(n_dev);
        let mut rho_i = Vec::with_capacity(n_dev);
        for d in 0..n_dev {
            let cells: Vec<(f64, f64)> = (0..n_origins)
                .filter_map(|o| Some((p.get(o, d)?, i.get(o, d)?)))
                .collect();
            let (sp, si) = cells
                .iter()
                .fold((0.0, 0.0), |(a, b), (x, y)| (a + x, b + y));
            let (qs, qi) = (sp / si, si / sp);
            let df = cells.len() as f64 - 1.0;
            let deviation = |dev: fn(f64, f64, f64, f64) -> f64| {
                (cells.len() >= 2).then(|| {
                    let ss: f64 = cells.iter().map(|&(x, y)| dev(x, y, qs, qi)).sum();
                    (ss / df).sqrt()
                })
            };
            rho_i.push(deviation(|x, y, qs, _| y * (x / y - qs).powi(2)));
            rho_p.push(deviation(|x, y, _, qi| x * (y / x - qi).powi(2)));
            q.push(qs);
            q_inverse.push(qi);
        }
        let rho_paid = interpolate_sigma(&rho_p, SigmaInterpolation::LogLinear);
        let rho_incurred = interpolate_sigma(&rho_i, SigmaInterpolation::LogLinear);

        // Residual pairs over the links whose sigma was estimated.
        let (mut pxy, mut pxx, mut ixy, mut ixx) = (0.0, 0.0, 0.0, 0.0);
        let (mut n_residuals_paid, mut n_residuals_incurred) = (0, 0);
        let (dp, di) = (
            &paid_fit.chain_ladder.development,
            &incurred_fit.chain_ladder.development,
        );
        let (fp, fi) = (paid_fit.chain_ladder.ldf(), incurred_fit.chain_ladder.ldf());
        for k in 0..n_dev - 1 {
            if dp.n_links[k] < 2 || di.n_links[k] < 2 {
                continue;
            }
            for o in 0..n_origins {
                let (Some(p0), Some(p1), Some(i0), Some(i1)) =
                    (p.get(o, k), p.get(o, k + 1), i.get(o, k), i.get(o, k + 1))
                else {
                    continue;
                };
                let paid_res = (p1 / p0 - fp[k]) / dp.sigma[k] * p0.sqrt();
                let incurred_res = (i1 / i0 - fi[k]) / di.sigma[k] * i0.sqrt();
                let q_inv_res = (i0 / p0 - 1.0 / q[k]) / rho_paid[k] * p0.sqrt();
                let q_res = (p0 / i0 - q[k]) / rho_incurred[k] * i0.sqrt();
                // Each column's regression drops its own pairs that are not
                // finite, as R's two `lm` fits do.
                if paid_res.is_finite() && q_inv_res.is_finite() {
                    pxy += q_inv_res * paid_res;
                    pxx += q_inv_res * q_inv_res;
                    n_residuals_paid += 1;
                }
                if incurred_res.is_finite() && q_res.is_finite() {
                    ixy += q_res * incurred_res;
                    ixx += q_res * q_res;
                    n_residuals_incurred += 1;
                }
            }
        }
        let slope = |xy: f64, xx: f64| if xx > 0.0 { xy / xx } else { 0.0 };

        let mut fit = MunichFit {
            origins: p.origins.clone(),
            development: p.ages.clone(),
            latest_position: paid_fit.chain_ladder.latest_position.clone(),
            latest_paid: paid_fit.chain_ladder.latest.clone(),
            latest_incurred: incurred_fit.chain_ladder.latest.clone(),
            ultimate_paid: Vec::new(),
            ultimate_incurred: Vec::new(),
            paid: paid_fit,
            incurred: incurred_fit,
            q,
            q_inverse,
            rho_paid,
            rho_incurred,
            lambda_paid: slope(pxy, pxx),
            lambda_incurred: slope(ixy, ixx),
            n_residuals_paid,
            n_residuals_incurred,
        };
        let (up, ui) = fit.project(fit.lambda_paid, fit.lambda_incurred);
        if let Some(o) = (0..n_origins).find(|&o| !(up[o].is_finite() && ui[o].is_finite())) {
            return Err(Error::Munich(format!(
                "the projection of origin {} is not finite: a ratio's deviation is zero at an \
                 age where the correction applies",
                fit.origins[o]
            )));
        }
        fit.ultimate_paid = up;
        fit.ultimate_incurred = ui;
        Ok(fit)
    }
}

/// Errors unless `dev` is Mack's volume-weighted model with every link
/// ratio and no selected pattern.
fn check_development(dev: &Development) -> Result<()> {
    let method = "the Munich chain ladder";
    dev.reject_geometric(method)?;
    let unsupported = |setting| {
        Err(Error::NotSupported {
            setting,
            method,
            reason: "its residuals are Mack's volume-weighted ones on every link ratio",
        })
    };
    if dev.average != Average::Volume {
        return unsupported("an average other than volume-weighted");
    }
    if dev.pattern.is_some() {
        return unsupported("a selected pattern");
    }
    if dev.has_exclusions() {
        return unsupported("n_periods, drop_high or drop_low");
    }
    Ok(())
}

/// Errors unless the paid and incurred segments have the same origins and
/// ages and are observed at the same cells, every one positive.
fn aligned(p: &Segment, i: &Segment, paid: &str, incurred: &str) -> Result<()> {
    if p.origins != i.origins || p.ages != i.ages {
        return Err(Error::Munich(format!(
            "{paid} and {incurred} must cover the same origins and ages: {paid} has {} origins \
             from {} and {} ages, {incurred} {} origins from {} and {} ages",
            p.n_origins, p.origins[0], p.n_dev, i.n_origins, i.origins[0], i.n_dev
        )));
    }
    for o in 0..p.n_origins {
        for d in 0..p.n_dev {
            let at = || format!("origin {} at age {}", p.origins[o], p.ages[d]);
            match (p.get(o, d), i.get(o, d)) {
                (Some(_), None) => {
                    return Err(Error::Munich(format!(
                        "{} has {paid} but no {incurred}: the two columns must be observed at \
                         the same cells",
                        at()
                    )));
                }
                (None, Some(_)) => {
                    return Err(Error::Munich(format!(
                        "{} has {incurred} but no {paid}: the two columns must be observed at \
                         the same cells",
                        at()
                    )));
                }
                (Some(x), Some(y)) if !(x > 0.0 && y > 0.0) => {
                    return Err(Error::Munich(format!(
                        "{} has {paid} {x} and {incurred} {y}: both must be positive, for the \
                         ratios between them",
                        at()
                    )));
                }
                _ => {}
            }
        }
    }
    Ok(())
}

impl MunichFit {
    /// The paid and incurred ultimates projected with slopes `lambda_p` and
    /// `lambda_i`: the fit's with its own, the two separate chain ladders
    /// with 0 and 0.
    pub(crate) fn project(&self, lambda_p: f64, lambda_i: f64) -> (Vec<f64>, Vec<f64>) {
        let n_dev = self.development.len();
        let (cp, ci) = (&self.paid.chain_ladder, &self.incurred.chain_ladder);
        let (fp, fi) = (cp.ldf(), ci.ldf());
        let (sp, si) = (&cp.development.sigma, &ci.development.sigma);
        // A step's factor and sigma, the tail's at the last; `None` sigma
        // is a step with no correction (no tail).
        let step = |k: usize, f: &[f64], s: &[f64], tail: &crate::TailFit| {
            if k + 1 < n_dev {
                (f[k], Some(s[k]))
            } else {
                (tail.factor, (tail.factor != 1.0).then_some(tail.sigma))
            }
        };
        let correction = |lambda: f64, sigma: Option<f64>, rho: f64, ratio: f64, centre: f64| {
            match sigma {
                Some(sigma) if lambda != 0.0 => {
                    let c = lambda * sigma / rho * (ratio - centre);
                    // R's `MunichChainLadder` takes a missing correction as 0.
                    if c.is_nan() { 0.0 } else { c }
                }
                _ => 0.0,
            }
        };
        let mut up = Vec::with_capacity(self.origins.len());
        let mut ui = Vec::with_capacity(self.origins.len());
        for o in 0..self.origins.len() {
            let (mut pv, mut iv) = (self.latest_paid[o], self.latest_incurred[o]);
            for k in self.latest_position[o]..n_dev {
                let (f_p, s_p) = step(k, fp, sp, &cp.tail);
                let (f_i, s_i) = step(k, fi, si, &ci.tail);
                let c_p = correction(lambda_p, s_p, self.rho_paid[k], iv / pv, self.q_inverse[k]);
                let c_i = correction(lambda_i, s_i, self.rho_incurred[k], pv / iv, self.q[k]);
                (pv, iv) = (pv * (f_p + c_p), iv * (f_i + c_i));
            }
            up.push(pv);
            ui.push(iv);
        }
        (up, ui)
    }

    /// Paid reserve per origin: the Munich paid ultimate minus the latest
    /// paid.
    pub fn reserve_paid(&self) -> Vec<f64> {
        diff(&self.ultimate_paid, &self.latest_paid)
    }

    /// Incurred IBNR per origin: the Munich incurred ultimate minus the
    /// latest incurred.
    pub fn ibnr_incurred(&self) -> Vec<f64> {
        diff(&self.ultimate_incurred, &self.latest_incurred)
    }

    /// Latest paid over latest incurred per origin.
    pub fn latest_ratio(&self) -> Vec<f64> {
        ratio(&self.latest_paid, &self.latest_incurred)
    }

    /// Paid ultimate over incurred ultimate per origin. Separate chain
    /// ladders leave it anywhere; the Munich projection draws it towards
    /// the oldest origins' ratio.
    pub fn ultimate_ratio(&self) -> Vec<f64> {
        ratio(&self.ultimate_paid, &self.ultimate_incurred)
    }

    /// Total Munich paid ultimate.
    pub fn total_ultimate_paid(&self) -> f64 {
        self.ultimate_paid.iter().sum()
    }

    /// Total Munich incurred ultimate.
    pub fn total_ultimate_incurred(&self) -> f64 {
        self.ultimate_incurred.iter().sum()
    }

    /// One row per origin: `latest_paid`, `latest_incurred`,
    /// `latest_ratio`, `ultimate_paid`, `ultimate_incurred`,
    /// `ultimate_ratio`, `reserve_paid` (paid ultimate minus latest paid),
    /// `ibnr_incurred` (incurred ultimate minus latest incurred), and the
    /// separate chain ladders' `chain_ladder_paid` and
    /// `chain_ladder_incurred`.
    pub fn to_long(&self) -> FitTable {
        FitTable {
            keys: Vec::new(),
            origin: Some(self.origins.clone()),
            age: None,
            values: self.origin_columns(),
        }
    }

    /// One row per development age: `q`, `q_inverse`, `rho_paid`,
    /// `rho_incurred`, and each column's Mack `ldf_paid`, `sigma_paid`,
    /// `ldf_incurred`, `sigma_incurred` (from this age to the next, NaN at
    /// the oldest age).
    pub fn development_table(&self) -> FitTable {
        FitTable {
            keys: Vec::new(),
            origin: None,
            age: Some(self.development.clone()),
            values: self.age_columns(),
        }
    }

    fn origin_columns(&self) -> Vec<(String, Vec<f64>)> {
        [
            ("latest_paid", self.latest_paid.clone()),
            ("latest_incurred", self.latest_incurred.clone()),
            ("latest_ratio", self.latest_ratio()),
            ("ultimate_paid", self.ultimate_paid.clone()),
            ("ultimate_incurred", self.ultimate_incurred.clone()),
            ("ultimate_ratio", self.ultimate_ratio()),
            ("reserve_paid", self.reserve_paid()),
            ("ibnr_incurred", self.ibnr_incurred()),
            ("chain_ladder_paid", self.paid.chain_ladder.ultimate.clone()),
            (
                "chain_ladder_incurred",
                self.incurred.chain_ladder.ultimate.clone(),
            ),
        ]
        .into_iter()
        .map(|(n, v)| (n.to_string(), v))
        .collect()
    }

    fn age_columns(&self) -> Vec<(String, Vec<f64>)> {
        let padded = |v: &[f64]| {
            let mut v = v.to_vec();
            v.push(f64::NAN);
            v
        };
        let (cp, ci) = (&self.paid.chain_ladder, &self.incurred.chain_ladder);
        [
            ("q", self.q.clone()),
            ("q_inverse", self.q_inverse.clone()),
            ("rho_paid", self.rho_paid.clone()),
            ("rho_incurred", self.rho_incurred.clone()),
            ("ldf_paid", padded(cp.ldf())),
            ("sigma_paid", padded(&cp.development.sigma)),
            ("ldf_incurred", padded(ci.ldf())),
            ("sigma_incurred", padded(&ci.development.sigma)),
        ]
        .into_iter()
        .map(|(n, v)| (n.to_string(), v))
        .collect()
    }

    fn total_columns(&self) -> Vec<(String, f64)> {
        let sum = |v: &[f64]| v.iter().sum::<f64>();
        let (lp, li) = (sum(&self.latest_paid), sum(&self.latest_incurred));
        let (up, ui) = (self.total_ultimate_paid(), self.total_ultimate_incurred());
        [
            ("latest_paid", lp),
            ("latest_incurred", li),
            ("latest_ratio", lp / li),
            ("ultimate_paid", up),
            ("ultimate_incurred", ui),
            ("ultimate_ratio", up / ui),
            ("reserve_paid", up - lp),
            ("ibnr_incurred", ui - li),
            ("chain_ladder_paid", self.paid.chain_ladder.total_ultimate()),
            (
                "chain_ladder_incurred",
                self.incurred.chain_ladder.total_ultimate(),
            ),
            ("lambda_paid", self.lambda_paid),
            ("lambda_incurred", self.lambda_incurred),
            ("n_residuals_paid", self.n_residuals_paid as f64),
            ("n_residuals_incurred", self.n_residuals_incurred as f64),
        ]
        .into_iter()
        .map(|(n, v)| (n.to_string(), v))
        .collect()
    }
}

/// `a - b`, element by element.
fn diff(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b).map(|(x, y)| x - y).collect()
}

/// `a / b`, element by element.
fn ratio(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b).map(|(x, y)| x / y).collect()
}

/// Appends each segment's `columns` row by row: one table of every
/// segment's rows, in segment order.
fn stacked(per_segment: impl Iterator<Item = Vec<(String, Vec<f64>)>>) -> Vec<(String, Vec<f64>)> {
    let mut values: Vec<(String, Vec<f64>)> = Vec::new();
    for columns in per_segment {
        if values.is_empty() {
            values = columns;
        } else {
            for ((_, all), (_, v)) in values.iter_mut().zip(columns) {
                all.extend(v);
            }
        }
    }
    values
}

impl SegmentFits<MunichFit> {
    /// One row per segment × origin, the columns of
    /// [`MunichFit::to_long`].
    pub fn munich_long(&self) -> FitTable {
        FitTable {
            keys: self.key_columns(|s| self.fits[s].origins.len()),
            origin: Some(
                self.fits
                    .iter()
                    .flat_map(|f| f.origins.iter().copied())
                    .collect(),
            ),
            age: None,
            values: stacked(self.fits.iter().map(MunichFit::origin_columns)),
        }
    }

    /// One row per segment × development age, the columns of
    /// [`MunichFit::development_table`].
    pub fn munich_development_table(&self) -> FitTable {
        FitTable {
            keys: self.key_columns(|s| self.fits[s].development.len()),
            origin: None,
            age: Some(
                self.fits
                    .iter()
                    .flat_map(|f| f.development.iter().copied())
                    .collect(),
            ),
            values: stacked(self.fits.iter().map(MunichFit::age_columns)),
        }
    }

    /// One row per segment: the totals of the per-origin columns (the
    /// ratios of the totals), `lambda_paid`, `lambda_incurred`,
    /// `n_residuals_paid` and `n_residuals_incurred`.
    pub fn munich_totals(&self) -> FitTable {
        FitTable {
            keys: self.key_columns(|_| 1),
            origin: None,
            age: None,
            values: stacked(self.fits.iter().map(|f| {
                f.total_columns()
                    .into_iter()
                    .map(|(n, v)| (n, vec![v]))
                    .collect()
            })),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chain_ladder::ChainLadder;
    use crate::triangle::{DevelopmentColumn, Long};
    use prospicio_core::{Grain, Month};

    /// Quarg and Mack's (2004) example, R ChainLadder's `MCLpaid` and
    /// `MCLincurred`.
    pub(crate) const PAID: [&[f64]; 7] = [
        &[576.0, 1804.0, 1970.0, 2024.0, 2074.0, 2102.0, 2131.0],
        &[866.0, 1948.0, 2162.0, 2232.0, 2284.0, 2348.0],
        &[1412.0, 3758.0, 4252.0, 4416.0, 4494.0],
        &[2286.0, 5292.0, 5724.0, 5850.0],
        &[1868.0, 3778.0, 4648.0],
        &[1442.0, 4010.0],
        &[2044.0],
    ];
    pub(crate) const INCURRED: [&[f64]; 7] = [
        &[978.0, 2104.0, 2134.0, 2144.0, 2174.0, 2182.0, 2174.0],
        &[1844.0, 2552.0, 2466.0, 2480.0, 2508.0, 2454.0],
        &[2904.0, 4354.0, 4698.0, 4600.0, 4644.0],
        &[3502.0, 5958.0, 6070.0, 6142.0],
        &[2812.0, 4882.0, 4852.0],
        &[2642.0, 4406.0],
        &[5022.0],
    ];

    fn pair(paid: &[&[f64]], incurred: &[&[f64]]) -> Triangle {
        let (mut origin, mut ages, mut p, mut i) = (vec![], vec![], vec![], vec![]);
        for (k, (rp, ri)) in paid.iter().zip(incurred).enumerate() {
            // NaN marks a cell one column does not observe.
            for d in 0..rp.len().max(ri.len()) {
                origin.push(Month::january(2001 + k as i32));
                ages.push(12 * (d as Lag + 1));
                p.push(rp.get(d).copied().unwrap_or(f64::NAN));
                i.push(ri.get(d).copied().unwrap_or(f64::NAN));
            }
        }
        Triangle::from_long(&Long {
            keys: &[],
            origin: &origin,
            development: DevelopmentColumn::Age(&ages),
            values: &[("paid", &p), ("incurred", &i)],
            origin_grain: Grain::Year,
            development_grain: Grain::Year,
            cumulative: true,
        })
        .unwrap()
    }

    fn mack_sigma() -> Development {
        Development {
            sigma_interpolation: SigmaInterpolation::Mack,
            ..Default::default()
        }
    }

    fn close(got: f64, want: f64, tol: f64) {
        assert!((got - want).abs() <= tol, "got {got}, want {want}");
    }

    #[test]
    fn mcl_matches_r() {
        // R ChainLadder 0.2.21: MunichChainLadder(MCLpaid, MCLincurred,
        // est.sigmaP = "Mack", est.sigmaI = "Mack"), the same as its
        // default call, which falls back to Mack's sigma on this data.
        let fit = MunichChainLadder {
            paid: mack_sigma(),
            incurred: mack_sigma(),
            ..Default::default()
        }
        .fit(&pair(&PAID, &INCURRED), "paid", "incurred")
        .unwrap();
        close(fit.lambda_paid, 0.636_021_466_425_142, 1e-12);
        close(fit.lambda_incurred, 0.436_187_132_012_033, 1e-12);
        close(fit.total_ultimate_paid(), 32_121.496_994_989_8, 1e-8);
        close(fit.total_ultimate_incurred(), 32_719.512_501_042_9, 1e-8);
        close(fit.ultimate_paid[6], 7_504.575_860_343_22, 1e-8);
        close(fit.ultimate_incurred[6], 7_655.377_611_105_52, 1e-8);
        assert_eq!((fit.n_residuals_paid, fit.n_residuals_incurred), (20, 20));
    }

    #[test]
    fn a_zero_sigma_in_one_column_keeps_the_other_columns_residuals() {
        // Incurred flat from 60 to 72 months in both origins observed
        // there, so incurred's sigma is 0 at that link and its residuals
        // there are 0/0. R ChainLadder 0.2.21 drops them from the
        // incurred regression only and keeps the paid ones:
        // MunichChainLadder(MCLpaid, I, est.sigmaP = "Mack",
        // est.sigmaI = "Mack") with I = MCLincurred, I[1, 6] <- I[1, 5],
        // I[2, 6] <- I[2, 5], I[1, 7] <- I[1, 6].
        let incurred: [&[f64]; 7] = [
            &[978.0, 2104.0, 2134.0, 2144.0, 2174.0, 2174.0, 2174.0],
            &[1844.0, 2552.0, 2466.0, 2480.0, 2508.0, 2508.0],
            INCURRED[2],
            INCURRED[3],
            INCURRED[4],
            INCURRED[5],
            INCURRED[6],
        ];
        let fit = MunichChainLadder {
            paid: mack_sigma(),
            incurred: mack_sigma(),
            ..Default::default()
        }
        .fit(&pair(&PAID, &incurred), "paid", "incurred")
        .unwrap();
        close(fit.lambda_paid, 0.636_021_466_425_142, 1e-12);
        close(fit.lambda_incurred, 0.413_992_067_837_447, 1e-12);
        close(fit.total_ultimate_paid(), 32_506.252_871_173_0, 1e-7);
        close(fit.total_ultimate_incurred(), 32_889.186_665_859_5, 1e-7);
        // nobs(m$lambdaP) and nobs(m$lambdaI).
        assert_eq!((fit.n_residuals_paid, fit.n_residuals_incurred), (20, 18));
    }

    #[test]
    fn zero_lambdas_are_the_separate_chain_ladders() {
        let tri = pair(&PAID, &INCURRED);
        let fit = MunichChainLadder::default()
            .fit(&tri, "paid", "incurred")
            .unwrap();
        let (up, ui) = fit.project(0.0, 0.0);
        let cp = ChainLadder::default().fit(&tri, "paid").unwrap();
        let ci = ChainLadder::default().fit(&tri, "incurred").unwrap();
        for o in 0..7 {
            close(up[o], cp.ultimate[o], 1e-9);
            close(ui[o], ci.ultimate[o], 1e-9);
        }
    }

    #[test]
    fn a_fixed_ratio_carries_no_information() {
        // Incurred twice paid: every ratio is 0.5, every rho 0, lambda 0,
        // and the projections are the separate chain ladders.
        let doubled: Vec<Vec<f64>> = PAID
            .iter()
            .map(|r| r.iter().map(|v| 2.0 * v).collect())
            .collect();
        let doubled: Vec<&[f64]> = doubled.iter().map(|r| r.as_slice()).collect();
        let tri = pair(&PAID, &doubled);
        let fit = MunichChainLadder::default()
            .fit(&tri, "paid", "incurred")
            .unwrap();
        assert_eq!((fit.lambda_paid, fit.lambda_incurred), (0.0, 0.0));
        let cp = ChainLadder::default().fit(&tri, "paid").unwrap();
        for o in 0..7 {
            close(fit.ultimate_paid[o], cp.ultimate[o], 1e-9);
            close(fit.ultimate_ratio()[o], 0.5, 1e-12);
        }
    }

    #[test]
    fn a_missing_incurred_cell_is_refused() {
        let short: Vec<&[f64]> = INCURRED
            .iter()
            .enumerate()
            .map(|(k, r)| if k == 2 { &r[..4] } else { *r })
            .collect();
        let err = MunichChainLadder::default()
            .fit(&pair(&PAID, &short), "paid", "incurred")
            .unwrap_err();
        let msg = err.to_string();
        assert!(
            matches!(err, Error::Munich(_)) && msg.contains("2003") && msg.contains("60"),
            "{msg}"
        );
    }

    #[test]
    fn unsupported_settings_are_refused() {
        let tri = pair(&PAID, &INCURRED);
        for dev in [
            Development {
                average: Average::Simple,
                ..Default::default()
            },
            Development {
                drop_high: 1,
                ..Default::default()
            },
        ] {
            let m = MunichChainLadder {
                incurred: dev,
                ..Default::default()
            };
            assert!(matches!(
                m.fit(&tri, "paid", "incurred"),
                Err(Error::NotSupported { .. })
            ));
        }
    }
}
