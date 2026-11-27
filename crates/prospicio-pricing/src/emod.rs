//! Experience modification: the split plan of workers compensation
//! (NCCI's Experience Rating Plan, [`SplitPlan`]) and the experience ratio
//! plan of general liability (ISO's Commercial General Liability
//! Experience Rating Plan, [`ExperienceRatioPlan`]). See
//! `docs/design/primary-rating.md`.
//!
//! # The split plan
//!
//! The plan compares a risk's own losses with what its payroll would be
//! expected to produce, giving the small, frequent losses (the **primary**
//! part of each claim, up to the split point) full weight and the large
//! ones (the **excess**) only the weighting value `W`. With expected losses
//! `E`, expected primary losses `E_p`, expected excess `E_e = E − E_p`,
//! actual primary `A_p`, actual excess `A_e` and the ballast `B`:
//!
//! ```text
//! mod = (A_p + W A_e + (1 − W) E_e + B) / (E + B)
//! ```
//!
//! which the bureau's worksheet writes as the adjusted actual losses over
//! the adjusted expected, `(A_p + W A_e + S) / (E_p + W E_e + S)`, with the
//! stabilizing value `S = (1 − W) E_e + B`.
//!
//! `E` is summed over the classes and years of the experience period as
//! the payroll in hundreds times the expected loss rate, and `E_p` as `E`
//! times the class's D-ratio. Each claim is first held to the per-claim
//! limit, the claims of one accident together to the per-accident limit;
//! its first `split_point` is then primary and the rest excess; and a
//! medical-only claim enters at a share of both (30% under the plan's
//! experience rating adjustment).
//!
//! The bureau's expected loss rates, D-ratios, weighting values, ballasts
//! and split points are licensed and change every filing, so they are
//! inputs here (`primary-rating.md` decision (b)). The exhibit keeps every
//! amount unrounded; the bureau's worksheet shows whole dollars and the
//! modification to two decimals.
//!
//! ```
//! use prospicio_pricing::emod::{Claim, ClassExposure, SplitPlan};
//!
//! let plan = SplitPlan::new(18_000.0).unwrap();
//! let exposures = [ClassExposure::new("8810", 3_000_000.0, 0.11, 0.35).unwrap()];
//! let claims = [
//!     Claim::new(25_000.0).unwrap(),
//!     Claim::new(2_000.0).unwrap().medical_only(),
//! ];
//! let exhibit = plan.modification(&exposures, &claims, 0.14, 5_000.0).unwrap();
//! assert_eq!(exhibit.expected, 3_300.0);
//! assert!((exhibit.actual_primary - (18_000.0 + 0.3 * 2_000.0)).abs() < 1e-9);
//! assert!((exhibit.actual_excess - 7_000.0).abs() < 1e-9);
//! assert!(exhibit.modification > 1.0);
//! ```
//!
//! # The experience ratio plan
//!
//! The general liability plan compares an actual experience ratio with
//! the expected one and credits or debits the difference at the
//! credibility `Z`:
//!
//! ```text
//! AER = (reported losses and ALAE + expected unreported) / subject loss cost
//! mod = 1 + Z (AER − EER) / EER
//! ```
//!
//! Each occurrence enters at its basic-limits loss plus its allocated loss
//! adjustment expense, held together to the maximum single limit (MSL).
//! The losses not yet reported for a year are expected as the year's
//! company subject loss cost times its detrend factor, the expected
//! experience ratio `EER` and the share of losses still unreported at the
//! evaluation. The subject loss costs, `EER`, MSL, credibility and
//! unreported shares are the bureau's tables, and inputs here.
//!
//! ```
//! use prospicio_pricing::emod::{ExperienceRatioPlan, LiabilityYear};
//!
//! let plan = ExperienceRatioPlan::new(50_000.0, 0.8, 0.3).unwrap();
//! let year = LiabilityYear::new(100_000.0, 0.25)
//!     .unwrap()
//!     .with_occurrence(70_000.0, 5_000.0)
//!     .unwrap();
//! let x = plan.modification(&[year]).unwrap();
//! // 50,000 reported (to the MSL) and 100,000 x 0.8 x 0.25 unreported.
//! assert!((x.actual_ratio - 0.7).abs() < 1e-12);
//! assert!((x.modification - (1.0 + 0.3 * (0.7 - 0.8) / 0.8)).abs() < 1e-12);
//! ```

use std::collections::HashMap;

use prospicio_core::{Error, Result};

/// The share of a medical-only claim that NCCI's experience rating
/// adjustment keeps.
pub const MEDICAL_ONLY_SHARE: f64 = 0.3;

/// The split plan's rules for each claim: the split point, the limits and
/// the share of a medical-only claim.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SplitPlan {
    /// The part of each claim, after the limits, that is primary.
    pub split_point: f64,
    /// The most of one claim that enters (the state's per-claim accident
    /// limitation); infinite for none.
    pub per_claim_limit: f64,
    /// The most of all the claims of one accident together that enters;
    /// infinite for none.
    pub per_accident_limit: f64,
    /// The share of a medical-only claim's primary and excess that enters.
    pub medical_only_share: f64,
}

fn invalid(name: &'static str, value: f64, reason: &'static str) -> Error {
    Error::InvalidParameter {
        name,
        value,
        reason,
    }
}

fn non_negative(name: &'static str, value: f64) -> Result<f64> {
    if value.is_finite() && value >= 0.0 {
        Ok(value)
    } else {
        Err(invalid(name, value, "must be finite and not negative"))
    }
}

fn limit(name: &'static str, value: f64) -> Result<f64> {
    if value > 0.0 && !value.is_nan() {
        Ok(value)
    } else {
        Err(invalid(name, value, "must be positive (infinite for none)"))
    }
}

fn share(name: &'static str, value: f64) -> Result<f64> {
    if (0.0..=1.0).contains(&value) {
        Ok(value)
    } else {
        Err(invalid(name, value, "must be between 0 and 1"))
    }
}

impl SplitPlan {
    /// A plan with this split point, no limits, and medical-only claims at
    /// [`MEDICAL_ONLY_SHARE`].
    pub fn new(split_point: f64) -> Result<Self> {
        Ok(Self {
            split_point: non_negative("split_point", split_point)?,
            per_claim_limit: f64::INFINITY,
            per_accident_limit: f64::INFINITY,
            medical_only_share: MEDICAL_ONLY_SHARE,
        })
    }

    /// This plan with a per-claim limit.
    pub fn with_per_claim_limit(self, limit_amount: f64) -> Result<Self> {
        Ok(Self {
            per_claim_limit: limit("per_claim_limit", limit_amount)?,
            ..self
        })
    }

    /// This plan with a limit on all the claims of one accident together.
    pub fn with_per_accident_limit(self, limit_amount: f64) -> Result<Self> {
        Ok(Self {
            per_accident_limit: limit("per_accident_limit", limit_amount)?,
            ..self
        })
    }

    /// This plan with another share for medical-only claims (1 enters
    /// them in full).
    pub fn with_medical_only_share(self, medical_only_share: f64) -> Result<Self> {
        Ok(Self {
            medical_only_share: share("medical_only_share", medical_only_share)?,
            ..self
        })
    }

    /// The modification of a risk with these exposures and claims over its
    /// experience period, at the weighting value `weight` and the ballast
    /// `ballast`, with every intermediate amount.
    ///
    /// Fails on a weighting value outside 0 to 1, a negative ballast, no
    /// expected losses and no ballast, or a plan, exposure or claim that is
    /// not valid.
    pub fn modification(
        &self,
        exposures: &[ClassExposure],
        claims: &[Claim],
        weight: f64,
        ballast: f64,
    ) -> Result<SplitExhibit> {
        self.check()?;
        let weight = share("weight", weight)?;
        let ballast = non_negative("ballast", ballast)?;
        let mut exposure_rows = Vec::with_capacity(exposures.len());
        for e in exposures {
            e.check()?;
            let expected = e.payroll / 100.0 * e.expected_loss_rate;
            exposure_rows.push(ExposureRow {
                expected,
                expected_primary: expected * e.d_ratio,
            });
        }
        for c in claims {
            non_negative("amount", c.amount)?;
        }
        let claim_rows = self.split_claims(claims);
        let expected: f64 = exposure_rows.iter().map(|r| r.expected).sum();
        let expected_primary: f64 = exposure_rows.iter().map(|r| r.expected_primary).sum();
        let expected_excess = expected - expected_primary;
        let actual_primary: f64 = claim_rows.iter().map(|r| r.primary).sum();
        let actual_excess: f64 = claim_rows.iter().map(|r| r.excess).sum();
        let stabilizing = (1.0 - weight) * expected_excess + ballast;
        let adjusted_actual = actual_primary + weight * actual_excess + stabilizing;
        let adjusted_expected = expected_primary + weight * expected_excess + stabilizing;
        if adjusted_expected <= 0.0 {
            return Err(invalid(
                "ballast",
                ballast,
                "the expected losses and the ballast are both zero",
            ));
        }
        Ok(SplitExhibit {
            exposures: exposure_rows,
            claims: claim_rows,
            expected,
            expected_primary,
            expected_excess,
            actual_primary,
            actual_excess,
            weight,
            ballast,
            stabilizing,
            adjusted_actual,
            adjusted_expected,
            modification: adjusted_actual / adjusted_expected,
        })
    }

    fn check(&self) -> Result<()> {
        non_negative("split_point", self.split_point)?;
        limit("per_claim_limit", self.per_claim_limit)?;
        limit("per_accident_limit", self.per_accident_limit)?;
        share("medical_only_share", self.medical_only_share)?;
        Ok(())
    }

    /// Each claim limited, split and, if medical-only, reduced.
    ///
    /// The per-accident limit takes what it removes from the excess of the
    /// accident's claims, in proportion to their excess, and only when the
    /// limit is below their primary losses from the primary too, in
    /// proportion, so that the accident's claims add up to the limit.
    fn split_claims(&self, claims: &[Claim]) -> Vec<ClaimRow> {
        let mut rows: Vec<ClaimRow> = claims
            .iter()
            .map(|c| {
                let limited = c.amount.min(self.per_claim_limit);
                let primary = limited.min(self.split_point);
                ClaimRow {
                    limited,
                    primary,
                    excess: limited - primary,
                }
            })
            .collect();
        if self.per_accident_limit.is_finite() {
            let mut accidents: HashMap<&str, Vec<usize>> = HashMap::new();
            for (i, c) in claims.iter().enumerate() {
                if let Some(a) = c.accident.as_deref() {
                    accidents.entry(a).or_default().push(i);
                }
            }
            for members in accidents.values() {
                self.limit_accident(&mut rows, members);
            }
            for (i, c) in claims.iter().enumerate() {
                if c.accident.is_none() {
                    self.limit_accident(&mut rows, &[i]);
                }
            }
        }
        for (row, c) in rows.iter_mut().zip(claims) {
            if c.medical_only {
                row.primary *= self.medical_only_share;
                row.excess *= self.medical_only_share;
            }
        }
        rows
    }

    fn limit_accident(&self, rows: &mut [ClaimRow], members: &[usize]) {
        let cap = self.per_accident_limit;
        let total: f64 = members.iter().map(|&i| rows[i].limited).sum();
        if total <= cap {
            return;
        }
        let primary: f64 = members.iter().map(|&i| rows[i].primary).sum();
        let excess = total - primary;
        for &i in members {
            let r = &mut rows[i];
            if cap >= primary {
                // Only the excess is cut: it keeps cap - primary in all.
                r.excess = if excess > 0.0 {
                    r.excess * (cap - primary) / excess
                } else {
                    0.0
                };
            } else {
                r.excess = 0.0;
                r.primary *= cap / primary;
            }
            r.limited = r.primary + r.excess;
        }
    }
}

/// One class's payroll in one year (or policy) of the experience period.
#[derive(Debug, Clone, PartialEq)]
pub struct ClassExposure {
    /// The class code, for the exhibit.
    pub class: String,
    /// The payroll, in money.
    pub payroll: f64,
    /// The expected loss rate per 100 of payroll.
    pub expected_loss_rate: f64,
    /// The D-ratio: the share of the expected losses that is primary.
    pub d_ratio: f64,
}

impl ClassExposure {
    /// A class's payroll with its expected loss rate (per 100 of payroll)
    /// and D-ratio.
    pub fn new(
        class: impl Into<String>,
        payroll: f64,
        expected_loss_rate: f64,
        d_ratio: f64,
    ) -> Result<Self> {
        let e = Self {
            class: class.into(),
            payroll,
            expected_loss_rate,
            d_ratio,
        };
        e.check()?;
        Ok(e)
    }

    fn check(&self) -> Result<()> {
        non_negative("payroll", self.payroll)?;
        non_negative("expected_loss_rate", self.expected_loss_rate)?;
        share("d_ratio", self.d_ratio)?;
        Ok(())
    }
}

/// One claim of the experience period, at its incurred amount.
#[derive(Debug, Clone, PartialEq)]
pub struct Claim {
    /// The incurred amount, indemnity and medical.
    pub amount: f64,
    /// Whether the claim is medical-only (no indemnity).
    pub medical_only: bool,
    /// The accident it arose from, for the per-accident limit; `None` for
    /// an accident of its own.
    pub accident: Option<String>,
}

impl Claim {
    /// A claim of this incurred amount, with indemnity, from an accident
    /// of its own.
    pub fn new(amount: f64) -> Result<Self> {
        Ok(Self {
            amount: non_negative("amount", amount)?,
            medical_only: false,
            accident: None,
        })
    }

    /// This claim as medical-only.
    pub fn medical_only(self) -> Self {
        Self {
            medical_only: true,
            ..self
        }
    }

    /// This claim as one of the claims of `accident`.
    pub fn in_accident(self, accident: impl Into<String>) -> Self {
        Self {
            accident: Some(accident.into()),
            ..self
        }
    }
}

/// A class row of [`SplitExhibit`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExposureRow {
    /// Payroll in hundreds times the expected loss rate.
    pub expected: f64,
    /// The expected losses times the D-ratio.
    pub expected_primary: f64,
}

/// A claim row of [`SplitExhibit`]: what of the claim enters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClaimRow {
    /// The claim after the per-claim and per-accident limits, before the
    /// medical-only share.
    pub limited: f64,
    /// The primary part that enters, after the medical-only share.
    pub primary: f64,
    /// The excess part that enters, after the medical-only share.
    pub excess: f64,
}

/// The split plan's worksheet: every sum the modification is made of.
#[derive(Debug, Clone, PartialEq)]
pub struct SplitExhibit {
    /// One row per exposure, in their order.
    pub exposures: Vec<ExposureRow>,
    /// One row per claim, in their order.
    pub claims: Vec<ClaimRow>,
    /// `E`, the expected losses.
    pub expected: f64,
    /// `E_p`, the expected primary losses.
    pub expected_primary: f64,
    /// `E_e = E − E_p`, the expected excess losses.
    pub expected_excess: f64,
    /// `A_p`, the actual primary losses.
    pub actual_primary: f64,
    /// `A_e`, the actual excess losses.
    pub actual_excess: f64,
    /// `W`, the weighting value.
    pub weight: f64,
    /// `B`, the ballast.
    pub ballast: f64,
    /// The stabilizing value `(1 − W) E_e + B`.
    pub stabilizing: f64,
    /// `A_p + W A_e` plus the stabilizing value.
    pub adjusted_actual: f64,
    /// `E_p + W E_e` plus the stabilizing value, which is `E + B`.
    pub adjusted_expected: f64,
    /// The adjusted actual over the adjusted expected, unrounded.
    pub modification: f64,
}

/// The general liability experience ratio plan: the maximum single limit,
/// the expected experience ratio and the credibility.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExperienceRatioPlan {
    /// The most of one occurrence's basic-limits loss and ALAE together
    /// that enters.
    pub maximum_single_limit: f64,
    /// The expected experience ratio `EER`.
    pub expected_ratio: f64,
    /// The credibility `Z`, from 0 to 1.
    pub credibility: f64,
}

impl ExperienceRatioPlan {
    /// A plan with this maximum single limit (infinite for none), expected
    /// experience ratio and credibility.
    pub fn new(maximum_single_limit: f64, expected_ratio: f64, credibility: f64) -> Result<Self> {
        let plan = Self {
            maximum_single_limit,
            expected_ratio,
            credibility,
        };
        plan.check()?;
        Ok(plan)
    }

    fn check(&self) -> Result<()> {
        limit("maximum_single_limit", self.maximum_single_limit)?;
        if !(self.expected_ratio.is_finite() && self.expected_ratio > 0.0) {
            return Err(invalid(
                "expected_ratio",
                self.expected_ratio,
                "must be positive and finite",
            ));
        }
        share("credibility", self.credibility)?;
        Ok(())
    }

    /// The modification of a risk with these years of experience, with
    /// every intermediate amount.
    ///
    /// Fails on a plan or year that is not valid, or when the years'
    /// subject loss costs add up to zero.
    pub fn modification(&self, years: &[LiabilityYear]) -> Result<RatioExhibit> {
        self.check()?;
        let mut rows = Vec::with_capacity(years.len());
        for y in years {
            y.check()?;
            let reported: f64 = y
                .occurrences
                .iter()
                .map(|&(loss, alae)| (loss + alae).min(self.maximum_single_limit))
                .sum();
            let expected_unreported =
                y.subject_loss_cost * y.detrend_factor * self.expected_ratio * y.unreported_share;
            rows.push(RatioYearRow {
                subject_loss_cost: y.subject_loss_cost,
                reported,
                expected_unreported,
            });
        }
        let subject_loss_cost: f64 = rows.iter().map(|r| r.subject_loss_cost).sum();
        if subject_loss_cost <= 0.0 {
            return Err(invalid(
                "subject_loss_cost",
                subject_loss_cost,
                "the years' subject loss costs add up to zero",
            ));
        }
        let reported: f64 = rows.iter().map(|r| r.reported).sum();
        let expected_unreported: f64 = rows.iter().map(|r| r.expected_unreported).sum();
        let actual_ratio = (reported + expected_unreported) / subject_loss_cost;
        let credit_debit =
            self.credibility * (actual_ratio - self.expected_ratio) / self.expected_ratio;
        Ok(RatioExhibit {
            years: rows,
            subject_loss_cost,
            reported,
            expected_unreported,
            actual_ratio,
            expected_ratio: self.expected_ratio,
            credibility: self.credibility,
            credit_debit,
            modification: 1.0 + credit_debit,
        })
    }
}

/// One policy year of a general liability risk's experience.
#[derive(Debug, Clone, PartialEq)]
pub struct LiabilityYear {
    /// The company subject basic-limits loss cost of the year.
    pub subject_loss_cost: f64,
    /// The share of the year's losses expected still unreported at the
    /// evaluation (the bureau's table, by the year's age).
    pub unreported_share: f64,
    /// The factor that brings the subject loss cost to the year's level for
    /// its expected unreported losses; 1 by default.
    pub detrend_factor: f64,
    /// Each occurrence's basic-limits loss and its ALAE.
    pub occurrences: Vec<(f64, f64)>,
}

impl LiabilityYear {
    /// A year of this subject loss cost and unreported share, no detrend,
    /// and no occurrences yet.
    pub fn new(subject_loss_cost: f64, unreported_share: f64) -> Result<Self> {
        let y = Self {
            subject_loss_cost,
            unreported_share,
            detrend_factor: 1.0,
            occurrences: Vec::new(),
        };
        y.check()?;
        Ok(y)
    }

    /// This year with a detrend factor.
    pub fn with_detrend_factor(self, detrend_factor: f64) -> Result<Self> {
        let y = Self {
            detrend_factor,
            ..self
        };
        y.check()?;
        Ok(y)
    }

    /// This year with one more occurrence: its basic-limits loss and ALAE.
    pub fn with_occurrence(mut self, loss: f64, alae: f64) -> Result<Self> {
        non_negative("loss", loss)?;
        non_negative("alae", alae)?;
        self.occurrences.push((loss, alae));
        Ok(self)
    }

    fn check(&self) -> Result<()> {
        non_negative("subject_loss_cost", self.subject_loss_cost)?;
        share("unreported_share", self.unreported_share)?;
        non_negative("detrend_factor", self.detrend_factor)?;
        for &(loss, alae) in &self.occurrences {
            non_negative("loss", loss)?;
            non_negative("alae", alae)?;
        }
        Ok(())
    }
}

/// A year row of [`RatioExhibit`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RatioYearRow {
    /// The year's company subject loss cost.
    pub subject_loss_cost: f64,
    /// Its occurrences' losses and ALAE, each held to the MSL.
    pub reported: f64,
    /// Subject loss cost x detrend x `EER` x unreported share.
    pub expected_unreported: f64,
}

/// The experience ratio plan's worksheet.
#[derive(Debug, Clone, PartialEq)]
pub struct RatioExhibit {
    /// One row per year, in their order.
    pub years: Vec<RatioYearRow>,
    /// The total subject loss cost.
    pub subject_loss_cost: f64,
    /// The total reported losses and ALAE, held to the MSL.
    pub reported: f64,
    /// The total expected unreported losses and ALAE.
    pub expected_unreported: f64,
    /// `AER`: reported and expected unreported over the subject loss cost.
    pub actual_ratio: f64,
    /// `EER`.
    pub expected_ratio: f64,
    /// `Z`.
    pub credibility: f64,
    /// `Z (AER − EER) / EER`, negative for a credit.
    pub credit_debit: f64,
    /// `1 + credit_debit`, unrounded.
    pub modification: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan() -> SplitPlan {
        SplitPlan::new(18_000.0)
            .unwrap()
            .with_per_claim_limit(250_000.0)
            .unwrap()
    }

    #[test]
    fn claims_split_at_the_split_point_after_the_limit() {
        // NCCI's ABCs of Experience Rating, Exhibit A.
        let claims = [500_000.0, 100_000.0, 5_000.0].map(|a| Claim::new(a).unwrap());
        let rows = plan().split_claims(&claims);
        let got: Vec<_> = rows
            .iter()
            .map(|r| (r.limited, r.primary, r.excess))
            .collect();
        assert_eq!(
            got,
            [
                (250_000.0, 18_000.0, 232_000.0),
                (100_000.0, 18_000.0, 82_000.0),
                (5_000.0, 5_000.0, 0.0)
            ]
        );
    }

    #[test]
    fn a_medical_only_claim_enters_at_its_share() {
        // Exhibit B: 20,000 medical-only gives 5,400 primary and 600 excess.
        let rows = plan().split_claims(&[Claim::new(20_000.0).unwrap().medical_only()]);
        assert!((rows[0].primary - 5_400.0).abs() < 1e-9);
        assert!((rows[0].excess - 600.0).abs() < 1e-9);
        assert_eq!(rows[0].limited, 20_000.0);
    }

    #[test]
    fn the_claims_of_one_accident_are_limited_together() {
        let p = plan().with_per_accident_limit(300_000.0).unwrap();
        let claims = [
            Claim::new(200_000.0).unwrap().in_accident("a"),
            Claim::new(150_000.0).unwrap().in_accident("a"),
            Claim::new(200_000.0).unwrap().in_accident("b"),
            Claim::new(260_000.0).unwrap(),
        ];
        let rows = p.split_claims(&claims);
        // Accident a: 350,000 to 300,000, the 50,000 out of the excess.
        assert!((rows[0].limited + rows[1].limited - 300_000.0).abs() < 1e-9);
        assert_eq!((rows[0].primary, rows[1].primary), (18_000.0, 18_000.0));
        let excess = rows[0].excess + rows[1].excess;
        assert!((excess - 264_000.0).abs() < 1e-9);
        assert!((rows[0].excess / rows[1].excess - 182.0 / 132.0).abs() < 1e-12);
        // Accident b and the claim on its own are under their limits.
        assert_eq!(rows[2].limited, 200_000.0);
        assert_eq!(rows[3].limited, 250_000.0);

        // A limit below the primaries cuts them in proportion.
        let tight = plan().with_per_accident_limit(20_000.0).unwrap();
        let rows = tight.split_claims(&claims[..2]);
        assert_eq!((rows[0].excess, rows[1].excess), (0.0, 0.0));
        assert!((rows[0].primary - 10_000.0).abs() < 1e-9);
    }

    #[test]
    fn the_modification_is_the_formula_and_one_at_expected() {
        let exposures = [
            ClassExposure::new("8288", 250_000.0, 3.50, 0.27).unwrap(),
            ClassExposure::new("8810", 3_000_000.0, 0.11, 0.35).unwrap(),
        ];
        let x = plan()
            .modification(&exposures, &[Claim::new(30_000.0).unwrap()], 0.14, 47_400.0)
            .unwrap();
        assert_eq!(x.exposures[0].expected, 8_750.0);
        assert!((x.exposures[0].expected_primary - 2_362.5).abs() < 1e-9);
        let (e, ep) = (8_750.0 + 3_300.0, 2_362.5 + 1_155.0);
        let ee = e - ep;
        assert!((x.expected - e).abs() < 1e-9 && (x.expected_excess - ee).abs() < 1e-9);
        let formula = (18_000.0 + 0.14 * 12_000.0 + 0.86 * ee + 47_400.0) / (e + 47_400.0);
        assert!((x.modification - formula).abs() < 1e-14);
        assert!((x.adjusted_expected - (e + 47_400.0)).abs() < 1e-9);

        // At full weight and no ballast a risk without losses gets 0.
        let none = plan().modification(&exposures, &[], 1.0, 0.0).unwrap();
        assert_eq!(none.modification, 0.0);
    }

    #[test]
    fn refuses_what_is_not_a_plan() {
        let e = [ClassExposure::new("8810", 100.0, 1.0, 0.3).unwrap()];
        assert!(plan().modification(&e, &[], 1.5, 0.0).is_err());
        assert!(plan().modification(&e, &[], 0.5, -1.0).is_err());
        assert!(plan().modification(&[], &[], 0.5, 0.0).is_err());
        assert!(SplitPlan::new(-1.0).is_err());
        assert!(plan().with_medical_only_share(1.2).is_err());
        assert!(plan().with_per_claim_limit(0.0).is_err());
        assert!(ClassExposure::new("x", 100.0, 1.0, 1.5).is_err());
        assert!(Claim::new(f64::NAN).is_err());
    }

    #[test]
    fn the_experience_ratio_plan_by_hand() {
        let plan = ExperienceRatioPlan::new(100_000.0, 0.75, 0.4).unwrap();
        let years = [
            LiabilityYear::new(200_000.0, 0.10)
                .unwrap()
                .with_detrend_factor(0.95)
                .unwrap()
                .with_occurrence(90_000.0, 20_000.0)
                .unwrap()
                .with_occurrence(10_000.0, 1_000.0)
                .unwrap(),
            LiabilityYear::new(220_000.0, 0.30)
                .unwrap()
                .with_occurrence(30_000.0, 0.0)
                .unwrap(),
            LiabilityYear::new(240_000.0, 0.60).unwrap(),
        ];
        let x = plan.modification(&years).unwrap();
        // 110,000 held to 100,000; 11,000; 30,000.
        assert_eq!(
            x.years.iter().map(|r| r.reported).collect::<Vec<_>>(),
            [111_000.0, 30_000.0, 0.0]
        );
        let unreported = [
            200_000.0 * 0.95 * 0.75 * 0.10,
            220_000.0 * 0.75 * 0.30,
            240_000.0 * 0.75 * 0.60,
        ];
        for (row, u) in x.years.iter().zip(unreported) {
            assert!((row.expected_unreported - u).abs() < 1e-9);
        }
        let aer = (141_000.0 + unreported.iter().sum::<f64>()) / 660_000.0;
        assert!((x.actual_ratio - aer).abs() < 1e-14);
        assert!((x.modification - (1.0 + 0.4 * (aer - 0.75) / 0.75)).abs() < 1e-14);
        assert!(x.credit_debit < 0.0);
    }

    #[test]
    fn at_the_expected_ratio_the_modification_is_one_and_no_credibility_gives_one() {
        // Reported losses that are exactly EER times the reported share.
        let plan = ExperienceRatioPlan::new(f64::INFINITY, 0.5, 0.8).unwrap();
        let year = LiabilityYear::new(100_000.0, 0.4)
            .unwrap()
            .with_occurrence(30_000.0, 0.0)
            .unwrap();
        let x = plan.modification(std::slice::from_ref(&year)).unwrap();
        assert!((x.modification - 1.0).abs() < 1e-14);
        let none = ExperienceRatioPlan::new(1e5, 0.5, 0.0).unwrap();
        assert_eq!(none.modification(&[year]).unwrap().modification, 1.0);
    }

    #[test]
    fn the_experience_ratio_plan_refuses_what_is_not_a_plan() {
        assert!(ExperienceRatioPlan::new(0.0, 0.5, 0.5).is_err());
        assert!(ExperienceRatioPlan::new(1e5, 0.0, 0.5).is_err());
        assert!(ExperienceRatioPlan::new(1e5, 0.5, 1.5).is_err());
        assert!(LiabilityYear::new(1.0, 1.5).is_err());
        assert!(
            LiabilityYear::new(1.0, 0.5)
                .unwrap()
                .with_occurrence(-1.0, 0.0)
                .is_err()
        );
        let plan = ExperienceRatioPlan::new(1e5, 0.5, 0.5).unwrap();
        assert!(plan.modification(&[]).is_err());
    }
}
