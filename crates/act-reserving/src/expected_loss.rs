//! The expected-loss family: expected loss ratio, Bornhuetter–Ferguson,
//! Benktander and Cape Cod (`docs/design/reserving-v02.md`, decisions 1
//! and 2), with the arithmetic of chainladder-python's estimators of the
//! same names.
//!
//! Each method takes an exposure column of the same triangle (premium,
//! say). An origin's exposure is that column's latest observed cumulative
//! value in the segment being fitted, as chainladder-python's examples pass
//! `sample_weight=premium.latest_diagonal`. With `q = 1 / cdf` the share of
//! the ultimate developed at the origin's latest age and an expected
//! ultimate `U0 = apriori * exposure`, the methods credit the latest value
//! `latest` and `U0` differently:
//!
//! * expected loss: `ultimate = U0`;
//! * Bornhuetter–Ferguson: `ultimate = latest + (1 - q) * U0`;
//! * Benktander: `U(k) = latest + (1 - q) * U(k - 1)`, `n_iters` times;
//! * Cape Cod: Bornhuetter–Ferguson with an apriori estimated from the
//!   triangle itself.

use act_core::Month;

use crate::chain_ladder::{ChainLadder, ChainLadderFit};
use crate::error::{Error, Result};
use crate::segments::{ReserveFit, SegmentFits, fit_each_with_exposure};
use crate::triangle::{Segment, Triangle};

/// The expected loss ratio method: each origin's ultimate is `apriori`
/// times its exposure, whatever has been observed.
///
/// The chain ladder is still fitted, for the development pattern the fit
/// reports, but does not move the ultimate.
///
/// ```
/// use act_reserving::{DevelopmentColumn, ExpectedLoss, Grain, Long, Month, Triangle};
///
/// let origin = [2020, 2020, 2021].map(Month::january);
/// let tri = Triangle::from_long(&Long {
///     keys: &[],
///     origin: &origin,
///     development: DevelopmentColumn::Age(&[12, 24, 12]),
///     values: &[
///         ("paid", &[100.0, 150.0, 200.0]),
///         ("premium", &[250.0, 250.0, 400.0]),
///     ],
///     origin_grain: Grain::Year,
///     development_grain: Grain::Year,
///     cumulative: true,
/// })?;
/// let fit = ExpectedLoss { apriori: 0.5, ..Default::default() }.fit(&tri, "paid", "premium")?;
/// assert_eq!(fit.ultimate, vec![125.0, 200.0]);
/// assert_eq!(fit.reserves(), vec![-25.0, 0.0]);
/// # Ok::<(), act_reserving::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExpectedLoss {
    /// Expected loss ratio: the ultimate per unit of exposure.
    pub apriori: f64,
    /// The development pattern; it does not affect this method's ultimate.
    pub chain_ladder: ChainLadder,
}

/// The Bornhuetter–Ferguson method: the latest value plus the expected
/// loss `apriori * exposure` times the share still to develop, `1 - 1 / cdf`.
///
/// ```
/// use act_reserving::{BornhuetterFerguson, DevelopmentColumn, Grain, Long, Month, Triangle};
///
/// let origin = [2020, 2020, 2021].map(Month::january);
/// let tri = Triangle::from_long(&Long {
///     keys: &[],
///     origin: &origin,
///     development: DevelopmentColumn::Age(&[12, 24, 12]),
///     values: &[
///         ("paid", &[100.0, 150.0, 200.0]),
///         ("premium", &[250.0, 250.0, 400.0]),
///     ],
///     origin_grain: Grain::Year,
///     development_grain: Grain::Year,
///     cumulative: true,
/// })?;
/// // The 2021 origin is a third developed (cdf 1.5): 200 + (1/3) * 0.5 * 400.
/// let bf = BornhuetterFerguson { apriori: 0.5, ..Default::default() };
/// let fit = bf.fit(&tri, "paid", "premium")?;
/// assert_eq!(fit.ultimate[0], 150.0);
/// assert!((fit.ultimate[1] - (200.0 + 200.0 / 3.0)).abs() < 1e-12);
/// # Ok::<(), act_reserving::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BornhuetterFerguson {
    /// Expected loss ratio: the expected ultimate per unit of exposure.
    pub apriori: f64,
    /// How the development pattern is estimated.
    pub chain_ladder: ChainLadder,
}

/// The Benktander (iterated Bornhuetter–Ferguson) method: starting from
/// `U(0) = apriori * exposure`, `U(k) = latest + (1 - 1 / cdf) * U(k - 1)`
/// for `n_iters` steps. `n_iters = 0` is the expected loss method, 1 is
/// Bornhuetter–Ferguson, and many iterations approach the chain ladder.
///
/// The iterations stop early once the ultimates no longer change.
///
/// ```
/// use act_reserving::{Benktander, DevelopmentColumn, Grain, Long, Month, Triangle};
///
/// let origin = [2020, 2020, 2021].map(Month::january);
/// let tri = Triangle::from_long(&Long {
///     keys: &[],
///     origin: &origin,
///     development: DevelopmentColumn::Age(&[12, 24, 12]),
///     values: &[
///         ("paid", &[100.0, 150.0, 200.0]),
///         ("premium", &[250.0, 250.0, 400.0]),
///     ],
///     origin_grain: Grain::Year,
///     development_grain: Grain::Year,
///     cumulative: true,
/// })?;
/// // Many iterations give the chain ladder's 200 * 1.5.
/// let fit = Benktander { apriori: 0.5, n_iters: 200, ..Default::default() }
///     .fit(&tri, "paid", "premium")?;
/// assert!((fit.ultimate[1] - 300.0).abs() < 1e-9);
/// # Ok::<(), act_reserving::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Benktander {
    /// Expected loss ratio of the starting ultimate.
    pub apriori: f64,
    /// Number of Bornhuetter–Ferguson steps.
    pub n_iters: usize,
    /// How the development pattern is estimated.
    pub chain_ladder: ChainLadder,
}

/// The Cape Cod (Stanard–Bühlmann) method: Bornhuetter–Ferguson with each
/// origin's apriori estimated from the triangle, as chainladder-python's
/// `CapeCod`.
///
/// Origin `j`'s used-up exposure is `exposure[j] / cdf[j]`. Its latest
/// value is trended to the valuation by `(1 + trend)^(m[j] / 12)`, `m[j]`
/// the months from the end of the origin period to the triangle's
/// valuation. Origin `i`'s trended apriori is the sum over `j` of the
/// trended latest values weighted by `decay^|i - j|`, over the same
/// weighted sum of used-up exposures; dividing by `i`'s own trend factor
/// gives the apriori its Bornhuetter–Ferguson ultimate uses. With
/// `decay = 1` every origin shares one loss ratio; with `decay = 0` each
/// origin keeps its own and the method returns the chain ladder.
///
/// ```
/// use act_reserving::{CapeCod, DevelopmentColumn, Grain, Long, Month, Triangle};
///
/// let origin = [2020, 2020, 2021].map(Month::january);
/// let tri = Triangle::from_long(&Long {
///     keys: &[],
///     origin: &origin,
///     development: DevelopmentColumn::Age(&[12, 24, 12]),
///     values: &[
///         ("paid", &[100.0, 150.0, 200.0]),
///         ("premium", &[250.0, 250.0, 400.0]),
///     ],
///     origin_grain: Grain::Year,
///     development_grain: Grain::Year,
///     cumulative: true,
/// })?;
/// // Loss ratio (150 + 200) / (250 + 400 / 1.5) on both origins.
/// let fit = CapeCod::default().fit(&tri, "paid", "premium")?;
/// let elr = 350.0 / (250.0 + 400.0 / 1.5);
/// assert!((fit.expected_loss.apriori[1] - elr).abs() < 1e-12);
/// # Ok::<(), act_reserving::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CapeCod {
    /// Annual trend of the loss ratio; above -1.
    pub trend: f64,
    /// Weight of an origin `n` periods away, `decay^n`; from 0 to 1.
    pub decay: f64,
    /// How the development pattern is estimated.
    pub chain_ladder: ChainLadder,
}

impl Default for ExpectedLoss {
    fn default() -> Self {
        Self {
            apriori: 1.0,
            chain_ladder: ChainLadder::default(),
        }
    }
}

impl Default for BornhuetterFerguson {
    fn default() -> Self {
        Self {
            apriori: 1.0,
            chain_ladder: ChainLadder::default(),
        }
    }
}

impl Default for Benktander {
    fn default() -> Self {
        Self {
            apriori: 1.0,
            n_iters: 1,
            chain_ladder: ChainLadder::default(),
        }
    }
}

impl Default for CapeCod {
    fn default() -> Self {
        Self {
            trend: 0.0,
            decay: 1.0,
            chain_ladder: ChainLadder::default(),
        }
    }
}

/// A fitted expected-loss method (expected loss, Bornhuetter–Ferguson,
/// Benktander or Cape Cod), per origin.
#[derive(Debug, Clone, PartialEq)]
pub struct ExpectedLossFit {
    /// The development pattern, origins and latest values. Its `ultimate`
    /// is the chain ladder's, not this method's.
    pub chain_ladder: ChainLadderFit,
    /// Exposure per origin: the exposure column's latest observed value.
    pub exposure: Vec<f64>,
    /// Expected loss ratio applied per origin (for Cape Cod, the detrended
    /// estimate).
    pub apriori: Vec<f64>,
    /// This method's ultimate per origin.
    pub ultimate: Vec<f64>,
}

impl ExpectedLossFit {
    /// Reserve (ultimate minus latest) per origin, often labelled IBNR.
    pub fn reserves(&self) -> Vec<f64> {
        self.ultimate
            .iter()
            .zip(&self.chain_ladder.latest)
            .map(|(u, l)| u - l)
            .collect()
    }

    /// Total ultimate across origins.
    pub fn total_ultimate(&self) -> f64 {
        self.ultimate.iter().sum()
    }

    /// Total reserve across origins.
    pub fn total_reserve(&self) -> f64 {
        self.total_ultimate() - self.chain_ladder.latest.iter().sum::<f64>()
    }

    /// Expected ultimate per origin, `apriori * exposure`.
    pub fn expected_ultimate(&self) -> Vec<f64> {
        self.exposure
            .iter()
            .zip(&self.apriori)
            .map(|(e, a)| e * a)
            .collect()
    }
}

/// A fitted Cape Cod: the expected-loss fit with the detrended apriori, and
/// the trended apriori it came from (chainladder-python's `apriori_`).
#[derive(Debug, Clone, PartialEq)]
pub struct CapeCodFit {
    /// Ultimates, exposures and the detrended apriori per origin
    /// (chainladder-python's `detrended_apriori_`).
    pub expected_loss: ExpectedLossFit,
    /// Apriori per origin at the valuation's cost level, before detrending.
    pub trended_apriori: Vec<f64>,
}

impl CapeCodFit {
    /// Reserve (ultimate minus latest) per origin.
    pub fn reserves(&self) -> Vec<f64> {
        self.expected_loss.reserves()
    }

    /// Total ultimate across origins.
    pub fn total_ultimate(&self) -> f64 {
        self.expected_loss.total_ultimate()
    }

    /// Total reserve across origins.
    pub fn total_reserve(&self) -> f64 {
        self.expected_loss.total_reserve()
    }
}

/// Checks that `apriori` is finite and positive.
fn check_apriori(apriori: f64) -> Result<()> {
    if apriori.is_finite() && apriori > 0.0 {
        Ok(())
    } else {
        Err(Error::InvalidSetting {
            name: "apriori",
            value: apriori,
            expected: "a finite, positive loss ratio",
        })
    }
}

/// Exposure of each origin of `segment`: the latest observed value of the
/// same origin in `exposure` (the exposure column of the same index
/// position, named `column`), which must be finite and positive.
fn origin_exposure(segment: &Segment, exposure: &Segment, column: &str) -> Result<Vec<f64>> {
    (0..segment.n_origins)
        .map(|o| {
            let position = (segment.origin_offset + o)
                .checked_sub(exposure.origin_offset)
                .filter(|&e| e < exposure.n_origins);
            position
                .and_then(|e| exposure.latest(e).ok())
                .map(|(_, v)| v)
                .filter(|v| v.is_finite() && *v > 0.0)
                .ok_or_else(|| Error::InvalidExposure {
                    column: column.to_string(),
                    origin: segment.origins[o].to_string(),
                })
        })
        .collect()
}

/// The chain ladder of `segment` and the exposure of each of its origins.
fn prepare(
    chain_ladder: &ChainLadder,
    segment: &Segment,
    exposure: &Segment,
    column: &str,
) -> Result<(ChainLadderFit, Vec<f64>)> {
    let fit = chain_ladder.fit_segment(segment, &segment.ages)?;
    let exposure = origin_exposure(segment, exposure, column)?;
    Ok((fit, exposure))
}

/// Age-to-ultimate factor at each origin's latest age.
fn latest_cdf(fit: &ChainLadderFit) -> impl Iterator<Item = f64> + '_ {
    fit.latest_position.iter().map(|&d| fit.cdf[d])
}

/// `n_iters` Bornhuetter–Ferguson steps from `expected`:
/// `U(k) = latest + (1 - 1 / cdf) * U(k - 1)`, stopping early once an
/// origin's ultimate no longer changes.
fn benktander(fit: &ChainLadderFit, expected: &[f64], n_iters: usize) -> Vec<f64> {
    latest_cdf(fit)
        .zip(&fit.latest)
        .zip(expected)
        .map(|((cdf, &latest), &u0)| {
            let unreported = 1.0 - 1.0 / cdf;
            let mut u = u0;
            for _ in 0..n_iters {
                let next = latest + unreported * u;
                if next == u {
                    break;
                }
                u = next;
            }
            u
        })
        .collect()
}

impl ExpectedLoss {
    /// Fits loss `column` of a single-segment triangle with exposure from
    /// its `exposure` column.
    pub fn fit(
        &self,
        triangle: &Triangle,
        column: &str,
        exposure: &str,
    ) -> Result<ExpectedLossFit> {
        self.as_benktander().fit(triangle, column, exposure)
    }

    /// Fits `column` in every segment of `triangle`, each with its own
    /// exposure; see [`SegmentFits`] for the long tables. A failure names
    /// its segment.
    pub fn fit_segments(
        &self,
        triangle: &Triangle,
        column: &str,
        exposure: &str,
    ) -> Result<SegmentFits<ExpectedLossFit>> {
        self.as_benktander()
            .fit_segments(triangle, column, exposure)
    }

    fn as_benktander(&self) -> Benktander {
        Benktander {
            apriori: self.apriori,
            n_iters: 0,
            chain_ladder: self.chain_ladder,
        }
    }
}

impl BornhuetterFerguson {
    /// Fits loss `column` of a single-segment triangle with exposure from
    /// its `exposure` column.
    pub fn fit(
        &self,
        triangle: &Triangle,
        column: &str,
        exposure: &str,
    ) -> Result<ExpectedLossFit> {
        self.as_benktander().fit(triangle, column, exposure)
    }

    /// Fits `column` in every segment of `triangle`, each with its own
    /// exposure; see [`SegmentFits`] for the long tables. A failure names
    /// its segment.
    pub fn fit_segments(
        &self,
        triangle: &Triangle,
        column: &str,
        exposure: &str,
    ) -> Result<SegmentFits<ExpectedLossFit>> {
        self.as_benktander()
            .fit_segments(triangle, column, exposure)
    }

    fn as_benktander(&self) -> Benktander {
        Benktander {
            apriori: self.apriori,
            n_iters: 1,
            chain_ladder: self.chain_ladder,
        }
    }
}

impl Benktander {
    /// Fits loss `column` of a single-segment triangle with exposure from
    /// its `exposure` column.
    pub fn fit(
        &self,
        triangle: &Triangle,
        column: &str,
        exposure: &str,
    ) -> Result<ExpectedLossFit> {
        let segment = triangle.segment(column)?;
        self.fit_segment(&segment, &triangle.segment(exposure)?, exposure)
    }

    /// Fits `column` in every segment of `triangle`, each with its own
    /// exposure; see [`SegmentFits`] for the long tables. A failure names
    /// its segment.
    pub fn fit_segments(
        &self,
        triangle: &Triangle,
        column: &str,
        exposure: &str,
    ) -> Result<SegmentFits<ExpectedLossFit>> {
        fit_each_with_exposure(triangle, column, exposure, |s, e| {
            self.fit_segment(s, e, exposure)
        })
    }

    fn fit_segment(
        &self,
        segment: &Segment,
        exposure: &Segment,
        column: &str,
    ) -> Result<ExpectedLossFit> {
        check_apriori(self.apriori)?;
        let (chain_ladder, exposure) = prepare(&self.chain_ladder, segment, exposure, column)?;
        let apriori = vec![self.apriori; exposure.len()];
        let expected: Vec<f64> = exposure.iter().map(|e| e * self.apriori).collect();
        let ultimate = benktander(&chain_ladder, &expected, self.n_iters);
        Ok(ExpectedLossFit {
            chain_ladder,
            exposure,
            apriori,
            ultimate,
        })
    }
}

impl CapeCod {
    /// Fits loss `column` of a single-segment triangle with exposure from
    /// its `exposure` column. Trend runs to the triangle's valuation.
    pub fn fit(&self, triangle: &Triangle, column: &str, exposure: &str) -> Result<CapeCodFit> {
        let segment = triangle.segment(column)?;
        let exposure_segment = triangle.segment(exposure)?;
        self.fit_segment(&segment, &exposure_segment, exposure, triangle.valuation())
    }

    /// Fits `column` in every segment of `triangle`, each with its own
    /// exposure and apriori; see [`SegmentFits`] for the long tables. Trend
    /// runs to the triangle's valuation in every segment. A failure names
    /// its segment.
    pub fn fit_segments(
        &self,
        triangle: &Triangle,
        column: &str,
        exposure: &str,
    ) -> Result<SegmentFits<CapeCodFit>> {
        let valuation = triangle.valuation();
        fit_each_with_exposure(triangle, column, exposure, |s, e| {
            self.fit_segment(s, e, exposure, valuation)
        })
    }

    fn fit_segment(
        &self,
        segment: &Segment,
        exposure: &Segment,
        column: &str,
        valuation: Month,
    ) -> Result<CapeCodFit> {
        if !self.trend.is_finite() || self.trend <= -1.0 {
            return Err(Error::InvalidSetting {
                name: "trend",
                value: self.trend,
                expected: "a finite annual rate above -1",
            });
        }
        if !(0.0..=1.0).contains(&self.decay) {
            return Err(Error::InvalidSetting {
                name: "decay",
                value: self.decay,
                expected: "a weight from 0 to 1",
            });
        }
        let (chain_ladder, exposure) = prepare(&self.chain_ladder, segment, exposure, column)?;

        // chainladder-python's `_get_capecod_aprioris`, without on-leveling.
        let used_up: Vec<f64> = exposure
            .iter()
            .zip(latest_cdf(&chain_ladder))
            .map(|(e, cdf)| e / cdf)
            .collect();
        let trend_factor: Vec<f64> = chain_ladder
            .origins
            .iter()
            .map(|o| {
                let months = valuation.months_since(o.end()).max(0);
                (1.0 + self.trend).powf(months as f64 / 12.0)
            })
            .collect();
        let trended_loss_ratio: Vec<f64> = chain_ladder
            .latest
            .iter()
            .zip(&trend_factor)
            .zip(&used_up)
            .map(|((l, t), u)| l * t / u)
            .collect();
        let n = used_up.len();
        let trended_apriori: Vec<f64> = (0..n)
            .map(|i| {
                let (mut num, mut den) = (0.0, 0.0);
                for j in 0..n {
                    let w = used_up[j] * self.decay.powf(i.abs_diff(j) as f64);
                    num += w * trended_loss_ratio[j];
                    den += w;
                }
                num / den
            })
            .collect();
        let apriori: Vec<f64> = trended_apriori
            .iter()
            .zip(&trend_factor)
            .map(|(a, t)| a / t)
            .collect();
        let expected: Vec<f64> = exposure.iter().zip(&apriori).map(|(e, a)| e * a).collect();
        let ultimate = benktander(&chain_ladder, &expected, 1);
        Ok(CapeCodFit {
            expected_loss: ExpectedLossFit {
                chain_ladder,
                exposure,
                apriori,
                ultimate,
            },
            trended_apriori,
        })
    }
}

impl ReserveFit for ExpectedLossFit {
    fn chain_ladder(&self) -> &ChainLadderFit {
        &self.chain_ladder
    }

    fn ultimate(&self) -> &[f64] {
        &self.ultimate
    }

    fn origin_columns(&self) -> Vec<(&'static str, Vec<f64>)> {
        vec![
            ("exposure", self.exposure.clone()),
            ("apriori", self.apriori.clone()),
        ]
    }

    fn total_columns(&self) -> Vec<(&'static str, f64)> {
        vec![("exposure", self.exposure.iter().sum())]
    }
}

impl ReserveFit for CapeCodFit {
    fn chain_ladder(&self) -> &ChainLadderFit {
        &self.expected_loss.chain_ladder
    }

    fn ultimate(&self) -> &[f64] {
        &self.expected_loss.ultimate
    }

    fn origin_columns(&self) -> Vec<(&'static str, Vec<f64>)> {
        let mut columns = self.expected_loss.origin_columns();
        columns.push(("trended_apriori", self.trended_apriori.clone()));
        columns
    }

    fn total_columns(&self) -> Vec<(&'static str, f64)> {
        self.expected_loss.total_columns()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::development::{Average, Development};
    use crate::triangle::tests::RAA;
    use crate::triangle::{DevelopmentColumn, Long};
    use crate::{Grain, Label};

    fn close(got: f64, want: f64, tol: f64) {
        assert!((got - want).abs() <= tol, "got {got}, want {want}");
    }

    /// An annual triangle from rows of cumulative `paid`, with each origin's
    /// `premium` repeated on its row (NaN leaves the origin without one).
    fn with_premium(rows: &[&[f64]], premium: &[f64]) -> Triangle {
        let (mut origin, mut ages, mut paid, mut prem) = (vec![], vec![], vec![], vec![]);
        for (k, row) in rows.iter().enumerate() {
            for (d, &v) in row.iter().enumerate() {
                origin.push(Month::january(1981 + k as i32));
                ages.push(12 * (d as u32 + 1));
                paid.push(v);
                prem.push(premium[k]);
            }
        }
        Triangle::from_long(&Long {
            keys: &[],
            origin: &origin,
            development: DevelopmentColumn::Age(&ages),
            values: &[("paid", &paid), ("premium", &prem)],
            origin_grain: Grain::Year,
            development_grain: Grain::Year,
            cumulative: true,
        })
        .unwrap()
    }

    /// RAA with a premium proportional to each origin's chain-ladder
    /// ultimate, so every origin's chain-ladder loss ratio is `elr`.
    fn raa_at_loss_ratio(elr: f64) -> (Triangle, ChainLadderFit) {
        let plain = with_premium(&RAA, &[1.0; 10]);
        let cl = ChainLadder::default().fit(&plain, "paid").unwrap();
        let premium: Vec<f64> = cl.ultimate.iter().map(|u| u / elr).collect();
        (with_premium(&RAA, &premium), cl)
    }

    #[test]
    fn hand_computed_two_origins() {
        // ldf 1.5, so 1982 has 1 - 1/1.5 = 1/3 still to develop.
        let tri = with_premium(&[&[100.0, 150.0], &[200.0]], &[250.0, 400.0]);
        let el = ExpectedLoss {
            apriori: 0.5,
            ..Default::default()
        }
        .fit(&tri, "paid", "premium")
        .unwrap();
        assert_eq!(el.ultimate, [125.0, 200.0]);
        assert_eq!(el.exposure, [250.0, 400.0]);
        assert_eq!(el.expected_ultimate(), [125.0, 200.0]);
        let bf = BornhuetterFerguson {
            apriori: 0.5,
            ..Default::default()
        }
        .fit(&tri, "paid", "premium")
        .unwrap();
        close(bf.ultimate[1], 200.0 + 200.0 / 3.0, 1e-12);
        // Benktander, two steps: 200 + (1/3)(200 + (1/3) 200).
        let bk = Benktander {
            apriori: 0.5,
            n_iters: 2,
            ..Default::default()
        }
        .fit(&tri, "paid", "premium")
        .unwrap();
        close(bk.ultimate[1], 200.0 + (200.0 + 200.0 / 3.0) / 3.0, 1e-12);
        close(bk.total_reserve(), bk.ultimate[1] - 200.0, 1e-12);
    }

    #[test]
    fn bf_at_the_chain_ladder_loss_ratio_is_the_chain_ladder() {
        // With exposure = CL ultimate / 0.6, BF with apriori 0.6 credits
        // exactly the chain ladder's expected development.
        let (tri, cl) = raa_at_loss_ratio(0.6);
        let bf = BornhuetterFerguson {
            apriori: 0.6,
            ..Default::default()
        }
        .fit(&tri, "paid", "premium")
        .unwrap();
        for (b, c) in bf.ultimate.iter().zip(&cl.ultimate) {
            close(*b, *c, 1e-8 * c);
        }
        // Cape Cod recovers that loss ratio for every origin, whatever the
        // decay, without trend.
        for decay in [1.0, 0.75, 0.0] {
            let cc = CapeCod {
                decay,
                ..Default::default()
            }
            .fit(&tri, "paid", "premium")
            .unwrap();
            for a in &cc.expected_loss.apriori {
                close(*a, 0.6, 1e-12);
            }
            assert_eq!(cc.trended_apriori, cc.expected_loss.apriori);
        }
    }

    #[test]
    fn benktander_iterations() {
        let tri = with_premium(&RAA, &[20_000.0; 10]);
        let fit = |n_iters| {
            Benktander {
                apriori: 0.8,
                n_iters,
                ..Default::default()
            }
            .fit(&tri, "paid", "premium")
            .unwrap()
        };
        let el = ExpectedLoss {
            apriori: 0.8,
            ..Default::default()
        };
        assert_eq!(fit(0), el.fit(&tri, "paid", "premium").unwrap());
        let bf = BornhuetterFerguson {
            apriori: 0.8,
            ..Default::default()
        };
        assert_eq!(fit(1), bf.fit(&tri, "paid", "premium").unwrap());
        // Many iterations converge to the chain ladder (RAA 1990's cdf is
        // 14.4, so (1 - 1/14.4)^k shrinks slowly).
        let cl = ChainLadder::default().fit(&tri, "paid").unwrap();
        let far = fit(2_000);
        for (b, c) in far.ultimate.iter().zip(&cl.ultimate) {
            close(*b, *c, 1e-9 * c);
        }
        // A huge n_iters stops once nothing changes.
        assert_eq!(fit(usize::MAX).ultimate.len(), 10);
    }

    #[test]
    fn cape_cod_without_decay_is_the_chain_ladder() {
        // decay 0: each origin's apriori is its own chain-ladder loss ratio,
        // trended and detrended again.
        let tri = with_premium(&RAA, &[20_000.0; 10]);
        let cl = ChainLadder::default().fit(&tri, "paid").unwrap();
        let cc = CapeCod {
            trend: 0.05,
            decay: 0.0,
            ..Default::default()
        }
        .fit(&tri, "paid", "premium")
        .unwrap();
        for (u, c) in cc.expected_loss.ultimate.iter().zip(&cl.ultimate) {
            close(*u, *c, 1e-9 * c);
        }
        // 1981 is trended nine years to the 1990 valuation.
        close(
            cc.trended_apriori[0] / cc.expected_loss.apriori[0],
            1.05f64.powi(9),
            1e-12,
        );
        assert_eq!(cc.trended_apriori[9], cc.expected_loss.apriori[9]);
    }

    #[test]
    fn cape_cod_hand_computed_with_trend() {
        // q = (1, 2/3); used-up exposure (250, 400/1.5); 1981 trended one
        // year to the 1982 valuation.
        let tri = with_premium(&[&[100.0, 150.0], &[200.0]], &[250.0, 400.0]);
        let cc = CapeCod {
            trend: 0.1,
            decay: 0.5,
            ..Default::default()
        }
        .fit(&tri, "paid", "premium")
        .unwrap();
        let used = [250.0, 400.0 / 1.5];
        let lr = [150.0 * 1.1 / used[0], 200.0 / used[1]];
        let a0 = (used[0] * lr[0] + 0.5 * used[1] * lr[1]) / (used[0] + 0.5 * used[1]);
        let a1 = (0.5 * used[0] * lr[0] + used[1] * lr[1]) / (0.5 * used[0] + used[1]);
        close(cc.trended_apriori[0], a0, 1e-12);
        close(cc.trended_apriori[1], a1, 1e-12);
        close(cc.expected_loss.apriori[0], a0 / 1.1, 1e-12);
        close(cc.expected_loss.ultimate[1], 200.0 + 400.0 * a1 / 3.0, 1e-9);
        assert_eq!(cc.expected_loss.ultimate[0], 150.0);
    }

    #[test]
    fn simple_average_development() {
        let tri = with_premium(&RAA, &[20_000.0; 10]);
        let simple = ChainLadder {
            development: Development {
                average: Average::Simple,
                ..Default::default()
            },
            ..Default::default()
        };
        let cl = simple.fit(&tri, "paid").unwrap();
        let bf = BornhuetterFerguson {
            apriori: 0.7,
            chain_ladder: simple,
        }
        .fit(&tri, "paid", "premium")
        .unwrap();
        assert_eq!(bf.chain_ladder, cl);
        let q = 1.0 / cl.cdf[0];
        close(bf.ultimate[9], 2063.0 + (1.0 - q) * 14_000.0, 1e-9);
    }

    #[test]
    fn missing_or_bad_exposure_is_an_error() {
        let nan = f64::NAN;
        let tri = with_premium(&[&[100.0, 150.0], &[200.0]], &[250.0, nan]);
        let bf = BornhuetterFerguson::default();
        let err = Error::InvalidExposure {
            column: "premium".into(),
            origin: "1982".into(),
        };
        assert_eq!(bf.fit(&tri, "paid", "premium"), Err(err.clone()));
        assert_eq!(
            err.to_string(),
            "origin 1982 has no observed, finite, positive exposure in column premium"
        );
        let zero = with_premium(&[&[100.0, 150.0], &[200.0]], &[0.0, 400.0]);
        assert!(matches!(
            CapeCod::default().fit(&zero, "paid", "premium"),
            Err(Error::InvalidExposure { origin, .. }) if origin == "1981"
        ));
        assert_eq!(
            bf.fit(&zero, "paid", "exposure"),
            Err(Error::UnknownColumn("exposure".into()))
        );
    }

    #[test]
    fn rejects_bad_settings() {
        let tri = with_premium(&[&[100.0, 150.0], &[200.0]], &[250.0, 400.0]);
        let bad = |r: Result<ExpectedLossFit>| matches!(r, Err(Error::InvalidSetting { .. }));
        assert!(bad(BornhuetterFerguson {
            apriori: 0.0,
            ..Default::default()
        }
        .fit(&tri, "paid", "premium")));
        assert!(bad(ExpectedLoss {
            apriori: f64::NAN,
            ..Default::default()
        }
        .fit(&tri, "paid", "premium")));
        for (trend, decay) in [(-1.0, 1.0), (0.0, 1.5), (0.0, -0.1)] {
            assert!(matches!(
                CapeCod {
                    trend,
                    decay,
                    ..Default::default()
                }
                .fit(&tri, "paid", "premium"),
                Err(Error::InvalidSetting { .. })
            ));
        }
    }

    #[test]
    fn segments_use_their_own_exposure_and_ultimate() {
        // Two lines; Home has double the premium of Auto.
        let origin = [2020, 2020, 2021, 2020, 2020, 2021].map(Month::january);
        let tri = Triangle::from_long(&Long {
            keys: &[("lob", &["Auto", "Auto", "Auto", "Home", "Home", "Home"])],
            origin: &origin,
            development: DevelopmentColumn::Age(&[12, 24, 12, 12, 24, 12]),
            values: &[
                ("paid", &[100.0, 150.0, 200.0, 10.0, 20.0, 30.0]),
                ("premium", &[250.0, 250.0, 400.0, 500.0, 500.0, 800.0]),
            ],
            origin_grain: Grain::Year,
            development_grain: Grain::Year,
            cumulative: true,
        })
        .unwrap();
        let bf = BornhuetterFerguson {
            apriori: 0.5,
            ..Default::default()
        };
        let fits = bf.fit_segments(&tri, "paid", "premium").unwrap();
        let home = fits.get(&Label::new(["Home"])).unwrap();
        assert_eq!(home.exposure, [500.0, 800.0]);
        // Home's ldf is 2: 30 + (1/2) * 0.5 * 800.
        assert_eq!(home.ultimate, [20.0, 230.0]);
        let long = fits.to_long();
        assert_eq!(long.column("ultimate").unwrap()[3], 230.0);
        assert_eq!(long.column("reserve").unwrap()[3], 200.0);
        assert_eq!(long.column("apriori").unwrap(), [0.5; 4]);
        let totals = fits.totals();
        assert_eq!(totals.column("reserve").unwrap()[1], 200.0);
        assert_eq!(totals.column("exposure").unwrap(), [650.0, 1300.0]);
        close(fits.total_reserve(), 200.0 / 3.0 + 200.0, 1e-12);
        let cc = CapeCod::default()
            .fit_segments(&tri, "paid", "premium")
            .unwrap();
        assert!(cc.to_long().column("trended_apriori").is_some());
        // A failing segment is named.
        let missing = Triangle::from_long(&Long {
            keys: &[("lob", &["Auto", "Auto", "Auto", "Home", "Home", "Home"])],
            origin: &origin,
            development: DevelopmentColumn::Age(&[12, 24, 12, 12, 24, 12]),
            values: &[
                ("paid", &[100.0, 150.0, 200.0, 10.0, 20.0, 30.0]),
                ("premium", &[250.0, 250.0, 400.0, 500.0, 500.0, -1.0]),
            ],
            origin_grain: Grain::Year,
            development_grain: Grain::Year,
            cumulative: true,
        })
        .unwrap();
        assert_eq!(
            bf.fit_segments(&missing, "paid", "premium")
                .unwrap_err()
                .to_string(),
            "segment Home: origin 2021 has no observed, finite, positive exposure in column premium"
        );
    }
}
