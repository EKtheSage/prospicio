//! The underwriting basis of a pattern and `underwriting_exposure_rust`.

use super::*;

/// An underwriting basis as `underwriting_basis()` in R encodes it: a list
/// of `writing` (monthly shares), `policy_months` and `reading` (the
/// default Weibull when missing).
pub(crate) fn underwriting_basis(value: &Robj) -> Result<UnderwritingBasis> {
    let bad = || Error::Other("ldf_underwriting must be an underwriting_basis()".into());
    let list = List::try_from(value).map_err(|_| bad())?;
    let (mut writing, mut policy_months, mut reading) = (None, None, AnnualReading::default());
    for (name, value) in list.iter() {
        match name {
            "writing" => writing = value.as_real_vector(),
            "policy_months" => policy_months = value.as_real(),
            "reading" if !value.is_null() => {
                reading = annual_reading(value.as_str().ok_or_else(bad)?)?;
            }
            _ => {}
        }
    }
    match (writing, policy_months) {
        (Some(writing), Some(months)) => Ok(basis_of(writing, months)?.with_reading(reading)),
        _ => Err(bad()),
    }
}

/// The reading named `"weibull"`, `"loglogistic"` or `"ln-linear"`
/// (`reserving-v02.md`, decision 25).
fn annual_reading(name: &str) -> Result<AnnualReading> {
    AnnualReading::from_name(name)
        .ok_or_else(|| crate::unknown_name("reading", name, AnnualReading::NAMES))
}

/// An underwriting basis of `writing` by month and policies of `months`
/// months, a whole number.
fn basis_of(writing: Vec<f64>, months: f64) -> Result<UnderwritingBasis> {
    if !(months >= 1.0 && months.fract() == 0.0 && months <= f64::from(u32::MAX)) {
        return Err(Error::Other(
            "policy_months must be a whole number of months, at least 1".into(),
        ));
    }
    Ok(UnderwritingBasis::new(writing, months as u32))
}

/// The share of an underwriting period's exposure in each accident month
/// from its start (`UnderwritingBasis::exposure`), which also checks the
/// basis.
/// @noRd
#[extendr]
fn underwriting_exposure_rust(writing: Vec<f64>, policy_months: f64) -> Result<Vec<f64>> {
    basis_of(writing, policy_months)?.exposure().map_err(to_r)
}

extendr_module! {
    mod basis;
    fn underwriting_exposure_rust;
}
