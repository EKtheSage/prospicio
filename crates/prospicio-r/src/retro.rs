//! Aggregate lane: retrospective rating for the R `retro.R` API
//! (`docs/design/primary-rating.md`): the insurance charge and savings
//! (Table M, Table L) and the basic premium that balances a plan.

use extendr_api::prelude::*;
use extendr_api::{Error, Result};
use prospicio_pricing::retro::{InsuranceCharge, RetroPlan};

use crate::distributions::Grid;
use crate::to_r;

/// The table of `aggregate`, a `grid_distribution`'s pointer or a numeric
/// vector of totals; a Table L on `unlimited_expected` unless that is NaN.
fn table_of(aggregate: &Robj, unlimited_expected: f64) -> Result<InsuranceCharge> {
    let table = if let Ok(grid) = <&Grid>::try_from(aggregate) {
        InsuranceCharge::from_severity(grid.inner.clone())
    } else if let Some(losses) = aggregate.as_real_slice() {
        InsuranceCharge::from_sample(losses)
    } else {
        return Err(Error::Other(
            "the aggregate must be a grid_distribution or a numeric vector of losses".into(),
        ));
    }
    .map_err(to_r)?;
    if unlimited_expected.is_nan() {
        Ok(table)
    } else {
        table.table_l(unlimited_expected).map_err(to_r)
    }
}

/// The charge and savings at `entry_ratios` for R's `insurance_charge()`.
#[extendr]
fn insurance_charge_rust(
    aggregate: Robj,
    entry_ratios: &[f64],
    unlimited_expected: f64,
) -> Result<List> {
    let table = table_of(&aggregate, unlimited_expected)?;
    let rows = table.table(entry_ratios);
    Ok(list!(
        entry_ratio = entry_ratios.to_vec(),
        charge = rows.iter().map(|r| r.charge).collect::<Vec<f64>>(),
        savings = rows.iter().map(|r| r.savings).collect::<Vec<f64>>(),
        expected = table.expected(),
        loss_elimination = table.loss_elimination()
    ))
}

/// The balanced plan for R's `retro_balance()`.
#[extendr]
#[allow(clippy::too_many_arguments)]
fn retro_balance_rust(
    aggregate: Robj,
    unlimited_expected: f64,
    expected_loss: f64,
    expense: f64,
    loss_conversion: f64,
    tax_multiplier: f64,
    minimum: f64,
    maximum: f64,
) -> Result<List> {
    let table = table_of(&aggregate, unlimited_expected)?;
    let plan = RetroPlan::new(
        expected_loss,
        expense,
        loss_conversion,
        tax_multiplier,
        minimum,
        maximum,
    )
    .map_err(to_r)?;
    let b = plan.balance(&table).map_err(to_r)?;
    Ok(list!(
        basic = b.basic,
        entry_ratio_minimum = b.entry_ratio_minimum,
        entry_ratio_maximum = b.entry_ratio_maximum,
        net_insurance_charge = b.net_insurance_charge,
        expected_premium = b.expected_premium
    ))
}

extendr_module! {
    mod retro;
    fn insurance_charge_rust;
    fn retro_balance_rust;
}
