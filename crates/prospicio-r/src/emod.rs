//! Aggregate lane: the experience modification of the workers
//! compensation split plan for the R `emod.R` API
//! (`docs/design/primary-rating.md`). A claim's accident crosses as a
//! string, empty for an accident of its own, and medical-only as 0 or 1.

use extendr_api::prelude::*;
use extendr_api::{Error, Result};
use prospicio_pricing::emod::{Claim, ClassExposure, SplitPlan};

use crate::to_r;

/// The worksheet for R's `split_plan_modification()`.
#[extendr]
#[allow(clippy::too_many_arguments)]
fn split_plan_modification_rust(
    class_code: Vec<String>,
    payroll: &[f64],
    expected_loss_rate: &[f64],
    d_ratio: &[f64],
    amount: &[f64],
    medical_only: &[f64],
    accident: Vec<String>,
    weight: f64,
    ballast: f64,
    split_point: f64,
    per_claim_limit: f64,
    per_accident_limit: f64,
    medical_only_share: f64,
) -> Result<List> {
    let n = class_code.len();
    if payroll.len() != n || expected_loss_rate.len() != n || d_ratio.len() != n {
        return Err(Error::Other(
            "the exposures' class, payroll, expected_loss_rate and d_ratio columns differ in length"
                .into(),
        ));
    }
    let m = amount.len();
    if medical_only.len() != m || accident.len() != m {
        return Err(Error::Other(
            "the claims' amount, medical_only and accident columns differ in length".into(),
        ));
    }
    let plan = SplitPlan::new(split_point)
        .and_then(|p| p.with_per_claim_limit(per_claim_limit))
        .and_then(|p| p.with_per_accident_limit(per_accident_limit))
        .and_then(|p| p.with_medical_only_share(medical_only_share))
        .map_err(to_r)?;
    let exposures = (0..n)
        .map(|i| {
            ClassExposure::new(
                class_code[i].clone(),
                payroll[i],
                expected_loss_rate[i],
                d_ratio[i],
            )
        })
        .collect::<prospicio_core::Result<Vec<_>>>()
        .map_err(to_r)?;
    let claims = (0..m)
        .map(|i| {
            let mut c = Claim::new(amount[i])?;
            if medical_only[i] != 0.0 {
                c = c.medical_only();
            }
            if !accident[i].is_empty() {
                c = c.in_accident(accident[i].clone());
            }
            Ok(c)
        })
        .collect::<prospicio_core::Result<Vec<_>>>()
        .map_err(to_r)?;
    let x = plan
        .modification(&exposures, &claims, weight, ballast)
        .map_err(to_r)?;
    Ok(list!(
        modification = x.modification,
        expected = x.expected,
        expected_primary = x.expected_primary,
        expected_excess = x.expected_excess,
        actual_primary = x.actual_primary,
        actual_excess = x.actual_excess,
        weight = x.weight,
        ballast = x.ballast,
        stabilizing = x.stabilizing,
        adjusted_actual = x.adjusted_actual,
        adjusted_expected = x.adjusted_expected,
        class_expected = x.exposures.iter().map(|r| r.expected).collect::<Vec<f64>>(),
        class_expected_primary = x
            .exposures
            .iter()
            .map(|r| r.expected_primary)
            .collect::<Vec<f64>>(),
        claim_limited = x.claims.iter().map(|r| r.limited).collect::<Vec<f64>>(),
        claim_primary = x.claims.iter().map(|r| r.primary).collect::<Vec<f64>>(),
        claim_excess = x.claims.iter().map(|r| r.excess).collect::<Vec<f64>>()
    ))
}

extendr_module! {
    mod emod;
    fn split_plan_modification_rust;
}
