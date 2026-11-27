//! Python bindings for `prospicio_pricing::emod`: the experience
//! modifications of the workers compensation split plan and the general
//! liability experience ratio plan.

use prospicio_pricing::emod::{
    Claim, ClassExposure, ExperienceRatioPlan, LiabilityYear, MEDICAL_ONLY_SHARE, SplitPlan,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::to_py;

fn same_length(name: &str, got: usize, want: usize, of: &str) -> PyResult<()> {
    if got == want {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!(
            "{name} has {got} entries, {of} has {want}"
        )))
    }
}

/// The experience modification of the workers compensation split plan
/// (NCCI's Experience Rating Plan), with every sum of its worksheet.
///
/// ``mod = (A_p + W A_e + (1 - W) E_e + B) / (E + B)``: expected losses
/// ``E`` (payroll in hundreds times the expected loss rate, summed over
/// the class rows) and their primary part ``E_p`` (times the D-ratio),
/// ``E_e = E - E_p``, actual primary and excess losses ``A_p`` and ``A_e``,
/// the weighting value ``W`` and the ballast ``B``. Each claim is held to
/// the per-claim limit, the claims of one accident together to the
/// per-accident limit (the cut taken from their excess first), split at
/// the split point, and a medical-only claim enters at
/// ``medical_only_share`` of both parts. The bureau's tables (expected loss
/// rates, D-ratios, weighting values, ballasts, split points) are inputs.
///
/// Parameters
/// ----------
/// class_code : list of str
///     One entry per class row of the experience period (a class in a
///     year or policy).
/// payroll, expected_loss_rate, d_ratio : list of float
///     Per class row: the payroll, the expected loss rate per 100 of
///     payroll, and the D-ratio.
/// amount : list of float
///     The incurred amount of each claim.
/// weight : float
///     The weighting value ``W``, from 0 to 1.
/// ballast : float
///     The ballast ``B``.
/// split_point : float
///     The part of each claim that is primary.
/// medical_only : list of bool, optional
///     Per claim, whether it is medical-only; none are by default.
/// accident : list of str or None, optional
///     Per claim, the accident it arose from, for the per-accident limit;
///     ``None`` for an accident of its own.
/// per_claim_limit, per_accident_limit : float, optional
///     No limit by default.
/// medical_only_share : float, default 0.3
///
/// Returns
/// -------
/// dict
///     ``modification`` (unrounded; the bureau shows two decimals),
///     ``expected``, ``expected_primary``, ``expected_excess``,
///     ``actual_primary``, ``actual_excess``, ``weight``, ``ballast``,
///     ``stabilizing`` (``(1 - W) E_e + B``), ``adjusted_actual``,
///     ``adjusted_expected``, and the rows: ``classes`` (``class_code``,
///     ``expected``, ``expected_primary``) and ``claims`` (``amount``,
///     ``limited``, ``primary``, ``excess``, what enters after the
///     limits and the medical-only share).
///
/// Raises
/// ------
/// ValueError
///     For columns of different lengths, a weight outside 0 to 1, a
///     negative amount, payroll, rate or ballast, a D-ratio outside 0 to 1,
///     or no expected losses and no ballast.
///
/// Examples
/// --------
/// >>> from prospicio.pricing import split_plan_modification
/// >>> x = split_plan_modification(["8810"], [3_000_000.0], [0.11], [0.35],
/// ...                             [25_000.0, 2_000.0], weight=0.14,
/// ...                             ballast=5_000.0, split_point=18_000.0,
/// ...                             medical_only=[False, True])
/// >>> x["expected"], x["actual_excess"], round(x["actual_primary"], 6)
/// (3300.0, 7000.0, 18600.0)
#[pyfunction]
#[pyo3(signature = (class_code, payroll, expected_loss_rate, d_ratio, amount, weight, ballast, split_point, medical_only = None, accident = None, per_claim_limit = None, per_accident_limit = None, medical_only_share = MEDICAL_ONLY_SHARE))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn split_plan_modification<'py>(
    py: Python<'py>,
    class_code: Vec<String>,
    payroll: Vec<f64>,
    expected_loss_rate: Vec<f64>,
    d_ratio: Vec<f64>,
    amount: Vec<f64>,
    weight: f64,
    ballast: f64,
    split_point: f64,
    medical_only: Option<Vec<bool>>,
    accident: Option<Vec<Option<String>>>,
    per_claim_limit: Option<f64>,
    per_accident_limit: Option<f64>,
    medical_only_share: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let n = class_code.len();
    same_length("payroll", payroll.len(), n, "class_code")?;
    same_length(
        "expected_loss_rate",
        expected_loss_rate.len(),
        n,
        "class_code",
    )?;
    same_length("d_ratio", d_ratio.len(), n, "class_code")?;
    let m = amount.len();
    if let Some(v) = &medical_only {
        same_length("medical_only", v.len(), m, "amount")?;
    }
    if let Some(v) = &accident {
        same_length("accident", v.len(), m, "amount")?;
    }
    let mut plan = SplitPlan::new(split_point)
        .and_then(|p| p.with_medical_only_share(medical_only_share))
        .map_err(to_py)?;
    if let Some(l) = per_claim_limit {
        plan = plan.with_per_claim_limit(l).map_err(to_py)?;
    }
    if let Some(l) = per_accident_limit {
        plan = plan.with_per_accident_limit(l).map_err(to_py)?;
    }
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
        .map_err(to_py)?;
    let claims = (0..m)
        .map(|i| {
            let mut c = Claim::new(amount[i])?;
            if medical_only.as_ref().is_some_and(|v| v[i]) {
                c = c.medical_only();
            }
            if let Some(a) = accident.as_ref().and_then(|v| v[i].clone()) {
                c = c.in_accident(a);
            }
            Ok(c)
        })
        .collect::<prospicio_core::Result<Vec<_>>>()
        .map_err(to_py)?;
    let x = plan
        .modification(&exposures, &claims, weight, ballast)
        .map_err(to_py)?;
    let d = PyDict::new(py);
    d.set_item("modification", x.modification)?;
    d.set_item("expected", x.expected)?;
    d.set_item("expected_primary", x.expected_primary)?;
    d.set_item("expected_excess", x.expected_excess)?;
    d.set_item("actual_primary", x.actual_primary)?;
    d.set_item("actual_excess", x.actual_excess)?;
    d.set_item("weight", x.weight)?;
    d.set_item("ballast", x.ballast)?;
    d.set_item("stabilizing", x.stabilizing)?;
    d.set_item("adjusted_actual", x.adjusted_actual)?;
    d.set_item("adjusted_expected", x.adjusted_expected)?;
    let classes = PyDict::new(py);
    classes.set_item("class_code", class_code)?;
    classes.set_item(
        "expected",
        x.exposures.iter().map(|r| r.expected).collect::<Vec<_>>(),
    )?;
    classes.set_item(
        "expected_primary",
        x.exposures
            .iter()
            .map(|r| r.expected_primary)
            .collect::<Vec<_>>(),
    )?;
    d.set_item("classes", classes)?;
    let rows = PyDict::new(py);
    rows.set_item("amount", amount)?;
    rows.set_item(
        "limited",
        x.claims.iter().map(|r| r.limited).collect::<Vec<_>>(),
    )?;
    rows.set_item(
        "primary",
        x.claims.iter().map(|r| r.primary).collect::<Vec<_>>(),
    )?;
    rows.set_item(
        "excess",
        x.claims.iter().map(|r| r.excess).collect::<Vec<_>>(),
    )?;
    d.set_item("claims", rows)?;
    Ok(d)
}

/// The experience modification of the general liability experience ratio
/// plan (ISO's Commercial General Liability Experience Rating Plan), with
/// every sum of its worksheet.
///
/// ``AER = (reported + expected unreported) / subject loss cost`` and
/// ``mod = 1 + Z (AER - EER) / EER``. Each occurrence enters at its
/// basic-limits loss plus ALAE, held together to the maximum single limit;
/// a year's expected unreported losses are its subject loss cost times its
/// detrend factor, ``EER`` and its unreported share. The bureau's tables
/// (subject loss costs, ``EER``, MSL, credibility, unreported shares) are
/// inputs.
///
/// Parameters
/// ----------
/// subject_loss_cost, unreported_share : list of float
///     Per year: the company subject basic-limits loss cost and the share
///     of its losses expected still unreported.
/// year, loss, alae : list
///     Per occurrence: the position of its year (from 0), its basic-limits
///     loss and its allocated loss adjustment expense.
/// maximum_single_limit : float
///     The MSL; ``float("inf")`` for none.
/// expected_ratio : float
///     ``EER``.
/// credibility : float
///     ``Z``, from 0 to 1.
/// detrend_factor : list of float, optional
///     Per year; 1 by default.
///
/// Returns
/// -------
/// dict
///     ``modification`` (unrounded), ``credit_debit``, ``actual_ratio``,
///     ``expected_ratio``, ``credibility``, ``subject_loss_cost``,
///     ``reported``, ``expected_unreported``, and ``years`` (columns
///     ``subject_loss_cost``, ``reported`` and ``expected_unreported``).
///
/// Raises
/// ------
/// ValueError
///     For columns of different lengths, a year position out of range, or
///     a plan or amount that is not valid.
///
/// Examples
/// --------
/// >>> from prospicio.pricing import experience_ratio_modification
/// >>> x = experience_ratio_modification([100_000.0], [0.25], [0], [70_000.0], [5_000.0],
/// ...                                   maximum_single_limit=50_000.0,
/// ...                                   expected_ratio=0.8, credibility=0.3)
/// >>> round(x["actual_ratio"], 12), round(x["modification"], 6)
/// (0.7, 0.9625)
#[pyfunction]
#[pyo3(signature = (subject_loss_cost, unreported_share, year, loss, alae, maximum_single_limit, expected_ratio, credibility, detrend_factor = None))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn experience_ratio_modification<'py>(
    py: Python<'py>,
    subject_loss_cost: Vec<f64>,
    unreported_share: Vec<f64>,
    year: Vec<usize>,
    loss: Vec<f64>,
    alae: Vec<f64>,
    maximum_single_limit: f64,
    expected_ratio: f64,
    credibility: f64,
    detrend_factor: Option<Vec<f64>>,
) -> PyResult<Bound<'py, PyDict>> {
    let n = subject_loss_cost.len();
    same_length(
        "unreported_share",
        unreported_share.len(),
        n,
        "subject_loss_cost",
    )?;
    if let Some(d) = &detrend_factor {
        same_length("detrend_factor", d.len(), n, "subject_loss_cost")?;
    }
    same_length("loss", loss.len(), year.len(), "year")?;
    same_length("alae", alae.len(), year.len(), "year")?;
    let plan = ExperienceRatioPlan::new(maximum_single_limit, expected_ratio, credibility)
        .map_err(to_py)?;
    let mut years = (0..n)
        .map(|i| {
            let y = LiabilityYear::new(subject_loss_cost[i], unreported_share[i])?;
            match &detrend_factor {
                Some(d) => y.with_detrend_factor(d[i]),
                None => Ok(y),
            }
        })
        .collect::<prospicio_core::Result<Vec<_>>>()
        .map_err(to_py)?;
    for (k, &i) in year.iter().enumerate() {
        let y = years.get_mut(i).ok_or_else(|| {
            PyValueError::new_err(format!(
                "year {i} of occurrence {k} is not a year: there are {n}, counted from 0"
            ))
        })?;
        *y = y.clone().with_occurrence(loss[k], alae[k]).map_err(to_py)?;
    }
    let x = plan.modification(&years).map_err(to_py)?;
    let d = PyDict::new(py);
    d.set_item("modification", x.modification)?;
    d.set_item("credit_debit", x.credit_debit)?;
    d.set_item("actual_ratio", x.actual_ratio)?;
    d.set_item("expected_ratio", x.expected_ratio)?;
    d.set_item("credibility", x.credibility)?;
    d.set_item("subject_loss_cost", x.subject_loss_cost)?;
    d.set_item("reported", x.reported)?;
    d.set_item("expected_unreported", x.expected_unreported)?;
    let rows = PyDict::new(py);
    rows.set_item("subject_loss_cost", subject_loss_cost)?;
    rows.set_item(
        "reported",
        x.years.iter().map(|r| r.reported).collect::<Vec<_>>(),
    )?;
    rows.set_item(
        "expected_unreported",
        x.years
            .iter()
            .map(|r| r.expected_unreported)
            .collect::<Vec<_>>(),
    )?;
    d.set_item("years", rows)?;
    Ok(d)
}
