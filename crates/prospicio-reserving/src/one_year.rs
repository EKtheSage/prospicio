//! The one-year view of reserve risk: the standard error of the claims
//! development result (CDR) of Merz and Wüthrich (2008), year by year over
//! the run-off as R ChainLadder's `CDR(MackChainLadder(x), dev = "all")`.
//!
//! The CDR of a calendar year is the change in the chain-ladder estimate of
//! the ultimate over that year, as one more diagonal is observed and the
//! factors are re-estimated. Its expected value is zero; its mean squared
//! error of prediction (MSEP) splits into process variance, from the
//! diagonal's own randomness, and parameter variance, from the change in
//! the estimated factors. Summed over every future calendar year, the
//! MSEPs give back Mack's MSEP of the ultimate (Merz and Wüthrich 2014).

use crate::error::{Error, Result};
use crate::mack::MackFit;
use prospicio_core::Period;

/// Standard errors of the claims development result per origin and in
/// total: of the next calendar year (the one-year view) and of every future
/// calendar year of the run-off, from a [`MackFit`].
///
/// Standard errors are square roots of Merz and Wüthrich's (2008) MSEP of
/// the CDR, in its linear approximation, which R ChainLadder's `CDR` also
/// uses. The total includes the covariance between origins that share
/// re-estimated factors.
///
/// ```
/// use prospicio_reserving::{DevelopmentColumn, Grain, Long, Mack, Month, Triangle};
///
/// let origin = [2020, 2020, 2020, 2020, 2021, 2021, 2021, 2022, 2022, 2023].map(Month::january);
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
/// let mack = Mack::default().fit(&tri, "paid")?;
/// let cdr = mack.claims_development_result()?;
/// // The oldest origin is fully developed; the next year's risk is part of
/// // the run-off risk, and the run-off adds up to Mack's.
/// assert_eq!(cdr.one_year_standard_error[0], 0.0);
/// assert!(cdr.total_one_year_standard_error < mack.total_standard_error);
/// assert!((cdr.total_run_off_standard_error() - mack.total_standard_error).abs() < 1e-9);
/// // Three future calendar years, one per remaining factor.
/// assert_eq!(cdr.by_calendar_year.len(), 3);
/// # Ok::<(), prospicio_reserving::Error>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct ClaimsDevelopmentResult {
    /// Origin periods, oldest first; every per-origin vector follows them.
    pub origins: Vec<Period>,
    /// Standard error of each origin's CDR in the next calendar year, R's
    /// `CDR(1)S.E.`.
    pub one_year_standard_error: Vec<f64>,
    /// Standard error of the total CDR in the next calendar year.
    pub total_one_year_standard_error: f64,
    /// Standard error of each origin's CDR in each future calendar year:
    /// element `[t][o]` is year `t + 1` (R's `CDR(t+1)S.E.`) of origin `o`,
    /// zero once the origin is fully developed. One year per age-to-age
    /// factor, so the youngest origin develops in every year.
    pub by_calendar_year: Vec<Vec<f64>>,
    /// Standard error of the total CDR in each future calendar year.
    pub total_by_calendar_year: Vec<f64>,
}

impl ClaimsDevelopmentResult {
    /// Standard error of each origin's full run-off: the square root of the
    /// sum of its MSEPs over the future calendar years. Equals Mack's
    /// standard error of the ultimate, as R's `Mack.S.E.` column shows.
    pub fn run_off_standard_error(&self) -> Vec<f64> {
        (0..self.origins.len())
            .map(|o| {
                self.by_calendar_year
                    .iter()
                    .map(|year| year[o].powi(2))
                    .sum::<f64>()
                    .sqrt()
            })
            .collect()
    }

    /// Standard error of the total full run-off; equals Mack's total
    /// standard error.
    pub fn total_run_off_standard_error(&self) -> f64 {
        self.total_by_calendar_year
            .iter()
            .map(|s| s * s)
            .sum::<f64>()
            .sqrt()
    }
}

impl MackFit {
    /// Merz and Wüthrich's (2008) claims development result: the standard
    /// error of each origin's and the total CDR in the next calendar year
    /// and in every later one, as R ChainLadder's
    /// `CDR(MackChainLadder(x), dev = "all")`.
    ///
    /// The formulas assume volume-weighted factors, no tail and a triangle
    /// whose latest values lie on one calendar diagonal, one new origin per
    /// period. Anything else is an error ([`Error::ClaimsDevelopment`]).
    ///
    /// An origin with an interior hole, a missing value before its latest,
    /// is accepted: each factor's volume is that of the pairs behind it, so
    /// the hole stays out now and next year and the run-off adds up to
    /// Mack's. R's `CDR` takes the volumes from the full triangle, the
    /// imputed cell included, and differs here.
    pub fn claims_development_result(&self) -> Result<ClaimsDevelopmentResult> {
        let cl = &self.chain_ladder;
        let dev = &cl.development;
        if dev.alpha != 1.0 {
            return Err(Error::ClaimsDevelopment(
                "needs volume-weighted factors (alpha = 1)",
            ));
        }
        // No tail: a factor of exactly 1 past the oldest age and no
        // estimated factor replaced within the triangle.
        let n_links = dev.ldf.len();
        if cl.tail.factor != 1.0 || cl.tail.attachment < n_links {
            return Err(Error::ClaimsDevelopment("needs no tail factor"));
        }
        // The factors the projection uses (the estimated ones, since no
        // tail replaced any).
        let ldf = cl.ldf();
        let n_origins = cl.origins.len();
        let regular = cl
            .latest_position
            .iter()
            .enumerate()
            .all(|(o, &p)| p == n_links.min(n_origins - 1 - o));
        if !regular {
            return Err(Error::ClaimsDevelopment(
                "the latest values must lie on one calendar diagonal, one new origin per period",
            ));
        }

        // Per factor k: sigma_k^2 / f_k^2, the volume S_k behind f_k, and
        // the share of the next estimate of f_k that comes from the latest
        // diagonal, C[I-k, k] / (S_k + C[I-k, k]) (Merz and Wüthrich's
        // alpha_k, R's `alpha`).
        let ratio: Vec<f64> = (0..n_links)
            .map(|k| (dev.sigma[k] / ldf[k]).powi(2))
            .collect();
        let volume = &dev.volume;
        let share: Vec<f64> = (0..n_links)
            .map(|k| {
                let diagonal = cl.latest[n_origins - 1 - k];
                diagonal / (volume[k] + diagonal)
            })
            .collect();
        // Product of (1 - share_i) for i in k + 1 - years ..= k (R's `y`):
        // the factors' estimates are diluted by each diagonal observed
        // before the CDR's year.
        let kept = |k: usize, years: usize| -> f64 {
            (k + 1 - years..=k).map(|i| 1.0 - share[i]).product()
        };

        let projections: Vec<Vec<f64>> = (0..n_origins).map(|o| cl.projection(o)).collect();
        let ultimate = &cl.ultimate;
        // Sum of the ultimates of the origins younger than each origin.
        let mut younger = vec![0.0; n_origins];
        for o in (0..n_origins.saturating_sub(1)).rev() {
            younger[o] = younger[o + 1] + ultimate[o + 1];
        }

        let mut by_calendar_year = Vec::with_capacity(n_links);
        let mut total_by_calendar_year = Vec::with_capacity(n_links);
        for year in 0..n_links {
            let mut msep = vec![0.0; n_origins];
            let mut total = 0.0;
            for o in 0..n_origins {
                let start = cl.latest_position[o];
                let j = start + year;
                if j >= n_links {
                    continue;
                }
                // Process variance of the factor developed this year, and
                // parameter variance from re-estimating this and every later
                // factor (R's CL_MSEPs).
                let c = projections[o][year];
                let process = c * cl.cdf[j].powi(2) * ratio[j];
                let parameter: f64 = (j..n_links)
                    .map(|l| {
                        let first = if l == j { 1.0 } else { share[l - year] };
                        kept(l, year) * first * ratio[l] / volume[l]
                    })
                    .sum();
                let u = ultimate[o];
                msep[o] = process + parameter * u * u;
                // The pairs (o, o') with o' younger share the older origin's
                // parameter term.
                total += process + parameter * u * (u + 2.0 * younger[o]);
            }
            by_calendar_year.push(msep.iter().map(|m| m.sqrt()).collect::<Vec<f64>>());
            total_by_calendar_year.push(total.sqrt());
        }

        Ok(ClaimsDevelopmentResult {
            origins: cl.origins.clone(),
            one_year_standard_error: by_calendar_year
                .first()
                .cloned()
                .unwrap_or_else(|| vec![0.0; n_origins]),
            total_one_year_standard_error: total_by_calendar_year.first().copied().unwrap_or(0.0),
            by_calendar_year,
            total_by_calendar_year,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::development::{Average, Development, SigmaInterpolation};
    use crate::triangle::tests::{annual, raa};
    use crate::{Mack, TailConstant, Triangle};

    fn close(got: f64, want: f64, tol: f64) {
        assert!((got - want).abs() <= tol, "got {got}, want {want}");
    }

    /// Merz and Wüthrich (2008), Table 2 (R ChainLadder::MW2008).
    fn mw2008() -> Triangle {
        annual(
            2001,
            &[
                &[
                    2_202_584.0,
                    3_210_449.0,
                    3_468_122.0,
                    3_545_070.0,
                    3_621_627.0,
                    3_644_636.0,
                    3_669_012.0,
                    3_674_511.0,
                    3_678_633.0,
                ],
                &[
                    2_350_650.0,
                    3_553_023.0,
                    3_783_846.0,
                    3_840_067.0,
                    3_865_187.0,
                    3_878_744.0,
                    3_898_281.0,
                    3_902_425.0,
                ],
                &[
                    2_321_885.0,
                    3_424_190.0,
                    3_700_876.0,
                    3_798_198.0,
                    3_854_755.0,
                    3_878_993.0,
                    3_898_825.0,
                ],
                &[
                    2_171_487.0,
                    3_165_274.0,
                    3_395_841.0,
                    3_466_453.0,
                    3_515_703.0,
                    3_548_422.0,
                ],
                &[
                    2_140_328.0,
                    3_157_079.0,
                    3_399_262.0,
                    3_500_520.0,
                    3_585_812.0,
                ],
                &[2_290_664.0, 3_338_197.0, 3_550_332.0, 3_641_036.0],
                &[2_148_216.0, 3_219_775.0, 3_428_335.0],
                &[2_143_728.0, 3_158_581.0],
                &[2_144_738.0],
            ],
        )
    }

    fn mack_sigma() -> Mack {
        Mack {
            development: Development {
                sigma_interpolation: SigmaInterpolation::Mack,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    #[test]
    fn mw2008_published_totals() {
        // Merz and Wüthrich (2008), Table 4, printed to the unit: reserves
        // 2'237'826, msep^(1/2) of the CDR 81'080, Mack 108'401, and the
        // youngest origin's 53'320 and 69'552. Mack's sigma rule fills the
        // last sigma, as the paper's (4.1).
        let mack = mack_sigma().fit(&mw2008(), "values").unwrap();
        let cdr = mack.claims_development_result().unwrap();
        close(mack.chain_ladder.total_reserve(), 2_237_826.0, 1.0);
        close(cdr.total_one_year_standard_error, 81_080.0, 1.0);
        close(cdr.total_run_off_standard_error(), 108_401.0, 1.0);
        close(cdr.one_year_standard_error[8], 53_320.0, 1.0);
        close(cdr.run_off_standard_error()[8], 69_552.0, 1.0);
    }

    #[test]
    fn mw2008_matches_r_by_calendar_year() {
        // R ChainLadder 0.2.21:
        // CDR(MackChainLadder(MW2008, est.sigma = "Mack"), dev = "all").
        let cdr = mack_sigma()
            .fit(&mw2008(), "values")
            .unwrap()
            .claims_development_result()
            .unwrap();
        let one_year = [
            0.0,
            566.17439488,
            1486.56034351,
            3923.09860757,
            9722.85976280,
            28442.62155591,
            20954.28697300,
            28119.31796274,
            53320.82104909,
        ];
        for (got, want) in cdr.one_year_standard_error.iter().zip(one_year) {
            close(*got, want, 1e-6);
        }
        let totals = [
            81080.54678704,
            52222.051578057,
            38517.494316141,
            29104.106607697,
            10109.002036812,
            3876.009319470,
            1281.302360689,
            399.458398221,
        ];
        assert_eq!(cdr.total_by_calendar_year.len(), totals.len());
        for (got, want) in cdr.total_by_calendar_year.iter().zip(totals) {
            close(*got, want, 1e-6);
        }
        // Origin 2007 develops in its third year after two others.
        close(cdr.by_calendar_year[2][6], 9340.472452171, 1e-6);
        assert_eq!(cdr.by_calendar_year[7][7], 0.0);
        close(cdr.total_run_off_standard_error(), 108_401.38745104, 1e-6);
    }

    #[test]
    fn raa_one_year_log_linear() {
        // R ChainLadder 0.2.21: CDR(MackChainLadder(RAA)).
        let mack = Mack::default().fit(&raa(), "values").unwrap();
        let cdr = mack.claims_development_result().unwrap();
        close(cdr.total_one_year_standard_error, 25_166.302526438, 1e-6);
        close(cdr.one_year_standard_error[9], 23_610.351930674, 1e-6);
        // The full run-off is Mack's standard error, per origin and in total.
        for (got, want) in cdr
            .run_off_standard_error()
            .iter()
            .zip(&mack.standard_error)
        {
            close(*got, *want, 1e-8 * want.max(1.0));
        }
        close(
            cdr.total_run_off_standard_error(),
            mack.total_standard_error,
            1e-8 * mack.total_standard_error,
        );
    }

    #[test]
    fn rejects_other_alpha_and_a_tail() {
        let simple = Mack {
            development: Development {
                average: Average::Simple,
                ..Default::default()
            },
            ..Default::default()
        }
        .fit(&raa(), "values")
        .unwrap();
        assert!(matches!(
            simple.claims_development_result(),
            Err(Error::ClaimsDevelopment(_))
        ));
        let tailed = Mack {
            tail: 1.05.into(),
            ..Default::default()
        }
        .fit(&raa(), "values")
        .unwrap();
        assert_eq!(
            tailed.claims_development_result(),
            Err(Error::ClaimsDevelopment("needs no tail factor"))
        );
        // A factor of 1 that replaces estimated factors is a tail too.
        let replaced = Mack {
            tail: TailConstant {
                factor: 1.0,
                attachment_age: Some(84),
                ..Default::default()
            }
            .into(),
            ..Default::default()
        }
        .fit(&raa(), "values")
        .unwrap();
        assert_eq!(replaced.chain_ladder.tail.factor, 1.0);
        assert!(
            replaced.chain_ladder.tail.attachment < replaced.chain_ladder.development.ldf.len()
        );
        assert_eq!(
            replaced.claims_development_result(),
            Err(Error::ClaimsDevelopment("needs no tail factor"))
        );
        // An explicit constant tail of 1 is no tail.
        let none = Mack {
            tail: 1.0.into(),
            ..Default::default()
        }
        .fit(&raa(), "values")
        .unwrap();
        assert!(none.claims_development_result().is_ok());
    }

    #[test]
    fn rejects_a_ragged_diagonal() {
        // No origin in the latest period: 2021 to 2023 end on the 2024
        // diagonal and 2023 is at age 24, so the positional shares would
        // be wrong.
        let t = annual(
            2020,
            &[
                &[100.0, 150.0, 165.0, 170.0],
                &[110.0, 170.0, 180.0, 185.0],
                &[120.0, 175.0, 190.0],
                &[130.0, 180.0],
            ],
        );
        let mack = Mack::default().fit(&t, "values").unwrap();
        assert!(matches!(
            mack.claims_development_result(),
            Err(Error::ClaimsDevelopment(_))
        ));
    }

    #[test]
    fn interior_hole_uses_the_pair_volumes() {
        // RAA without origin 1982 at age 48. Its latest is still on the
        // diagonal, so the shape check passes; 1982 stays out of the 36-48
        // and 48-60 factors now and next year, and the run-off adds up to
        // Mack's. R ChainLadder 0.2.21 MackChainLadder(r) with r[2, 4] <- NA
        // gives the Mack standard errors below. R's CDR on the same triangle
        // gives a one-year total of 23551.7570829886 and a run-off of
        // 24836.9792219666, short of its own Mack: it takes the factor
        // volumes from the full triangle, imputed cell included
        // (knowledge/references/r-chainladder-cdr.md).
        use crate::triangle::tests::RAA;
        use crate::triangle::{DevelopmentColumn, Long};
        use prospicio_core::{Grain, Lag, Month};

        let (mut origin, mut ages, mut values) = (Vec::new(), Vec::new(), Vec::new());
        for (o, row) in RAA.iter().enumerate() {
            for (d, &v) in row.iter().enumerate() {
                if (o, d) != (1, 3) {
                    origin.push(Month::january(1981 + o as i32));
                    ages.push(12 * (d as Lag + 1));
                    values.push(v);
                }
            }
        }
        let t = Triangle::from_long(&Long {
            keys: &[],
            origin: &origin,
            development: DevelopmentColumn::Age(&ages),
            values: &[("values", &values)],
            origin_grain: Grain::Year,
            development_grain: Grain::Year,
            cumulative: true,
        })
        .unwrap();
        let mack = Mack::default().fit(&t, "values").unwrap();
        let cdr = mack.claims_development_result().unwrap();
        let r_mack = [
            0.0,
            142.310257231384,
            591.891125677980,
            712.571908317654,
            1451.950171594476,
            1994.932606866756,
            2053.730986311831,
            3693.630055060053,
            5324.327029558024,
            23131.801249176013,
        ];
        for (got, want) in cdr.run_off_standard_error().iter().zip(r_mack) {
            close(*got, want, 1e-8 * want.max(1.0));
        }
        close(cdr.total_run_off_standard_error(), 24847.8292870902, 1e-7);
        // prospicio-reserving's own result with the pair volumes; R's differs.
        close(cdr.total_one_year_standard_error, 23557.357935729688, 1e-6);
    }

    #[test]
    fn more_origins_than_ages() {
        // A trapezoid: the two oldest origins are fully developed.
        let t = annual(
            2018,
            &[
                &[100.0, 150.0, 165.0, 170.0],
                &[105.0, 160.0, 170.0, 178.0],
                &[110.0, 170.0, 180.0, 186.0],
                &[120.0, 175.0, 190.0],
                &[130.0, 180.0],
                &[140.0],
            ],
        );
        let mack = Mack::default().fit(&t, "values").unwrap();
        let cdr = mack.claims_development_result().unwrap();
        assert_eq!(cdr.by_calendar_year.len(), 3);
        assert_eq!(cdr.one_year_standard_error[..3], [0.0; 3]);
        assert!(cdr.one_year_standard_error[3..].iter().all(|&s| s > 0.0));
        for (got, want) in cdr
            .run_off_standard_error()
            .iter()
            .zip(&mack.standard_error)
        {
            close(*got, *want, 1e-9 * want.max(1.0));
        }
        close(
            cdr.total_run_off_standard_error(),
            mack.total_standard_error,
            1e-9 * mack.total_standard_error,
        );
    }
}
