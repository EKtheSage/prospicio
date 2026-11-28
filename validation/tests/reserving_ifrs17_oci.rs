//! IFRS 17 with a current discount curve and the OCI option: Example 6 of
//! the IASB's Illustrative Examples (IE56 to IE71, a change in financial
//! assumptions and the exercise of discretion, a current rate beside the
//! locked-in one), and the OCI option's identities on a group whose curve
//! moves.
//!
//! Example 15 (IE152 to IE172, the allocation of finance income or expense)
//! could not be read from a cloud session (the page's text stops at IE70
//! through the fetch, and `www.ifrs.org` is refused to the shell), so its
//! parity waits (`reserving-ifrs17-measurement.md`, PR 2).

use prospicio_reserving::DiscountCurve;
use prospicio_reserving::ifrs17::{
    ActualCashFlows, CashFlowTiming, ComponentLine, ExpectedCashFlows, Group, GroupOptions,
    GroupPeriod, PeriodInputs,
};

/// The examples' figures are rounded at each step: one unit either way.
fn unit(name: &str, ours: f64, printed: f64) {
    assert!(
        (ours - printed).abs() <= 1.0,
        "{name}: {ours:.2} against the printed {printed}"
    );
}

fn flat(rate: f64) -> DiscountCurve {
    DiscountCurve::flat(rate).unwrap()
}

/// Claims at the end of each period, premiums at its start.
fn view(premiums: &[f64], claims: &[f64]) -> ExpectedCashFlows {
    ExpectedCashFlows::new(claims.len())
        .with_premiums(premiums.to_vec())
        .with_claims(claims.to_vec())
        .with_timing(
            CashFlowTiming::StartOfPeriod,
            CashFlowTiming::EndOfPeriod,
            CashFlowTiming::StartOfPeriod,
        )
}

fn paid(premiums: f64, claims: f64) -> ActualCashFlows {
    ActualCashFlows {
        premiums,
        claims,
        ..ActualCashFlows::default()
    }
}

fn line(p: &GroupPeriod, line: ComponentLine) -> (f64, f64, f64) {
    let r = p.component(line).unwrap();
    (r.present_value, r.risk_adjustment, r.csm)
}

/// Example 6: 200 contracts, a single premium of 3,000 in all, two deaths
/// at the end of each year and the account paid at maturity after three
/// years; the account is credited with the pool's return less 2 points.
/// Cash flows that vary with the returns are discounted at the current
/// rate (10% at recognition, the locked-in rate for the CSM's
/// adjustments); the CSM accretes at 4%, the rate for nominal cash flows
/// (B72(b)); coverage units are the contracts in force.
fn example_6() -> Group {
    let options = GroupOptions {
        coverage_units: Some(vec![200.0, 198.0, 196.0]),
        accretion_curve: Some(flat(0.04)),
        ..GroupOptions::default()
    };
    // IE68: deaths of 31, 33 and 35, and the maturity value of 3,346.
    let expected = view(&[3_000.0, 0.0, 0.0], &[31.0, 33.0, 35.0 + 3_346.0]);
    Group::recognise(expected, 30.0, flat(0.10), options).unwrap()
}

#[test]
fn example_6_initial_recognition_and_year_1() {
    let mut group = example_6();
    // IE65: present value of outflows 2,596, net (404), RA 30, CSM 374.
    let b = group.balances();
    unit("PV net", b.present_value, -404.0);
    unit("CSM", b.csm, 374.0);
    let year_1 = PeriodInputs::new(
        paid(3_000.0, 31.0),
        view(&[0.0, 0.0], &[33.0, 35.0 + 3_346.0]),
        20.0,
    )
    .with_risk_adjustment_released(10.0)
    .with_current_curve(flat(0.10));
    let p = group.roll_forward(&year_1).unwrap();
    // IE65 and IE71's opening: 2,824, RA 20, CSM 258.
    unit("PV", p.closing.present_value, 2_824.0);
    unit("CSM", p.closing.csm, 258.0);
    assert_eq!(p.finance_expenses_in_oci, 0.0);
}

#[test]
fn example_6_year_2_financial_assumptions_and_discretion() {
    let mut group = example_6();
    let year_1 = PeriodInputs::new(
        paid(3_000.0, 31.0),
        view(&[0.0, 0.0], &[33.0, 35.0 + 3_346.0]),
        20.0,
    )
    .with_risk_adjustment_released(10.0);
    group.roll_forward(&year_1).unwrap();
    // IE63 and IE69: the return falls to 7% in year 2 and is expected to stay
    // there; crediting at the pool return less 2 points gives 32 and 3,196
    // (3,228 in all, IE70) instead of 33 and 3,381: a change in financial
    // assumptions. The entity then credits 1 point more (discretion), 61 more
    // at maturity: 3,257 at the end of year 3.
    let revised = view(&[0.0, 0.0], &[32.0, 33.0 + 3_163.0]);
    let year_2 = PeriodInputs::new(paid(0.0, 32.0), view(&[0.0], &[33.0 + 3_224.0]), 10.0)
        .with_risk_adjustment_released(10.0)
        .with_financial_revision(revised)
        .with_current_curve(flat(0.07));
    let p = group.roll_forward(&year_2).unwrap();
    // IE71.
    let opening = line(&p, ComponentLine::Opening);
    unit("opening PV", opening.0, 2_824.0);
    unit("opening RA", opening.1, 20.0);
    unit("opening CSM", opening.2, 258.0);
    // Finance expenses: 197 on the present value (195 from the change in
    // financial assumptions and the unwind, 2 from measuring the discretion
    // at 7% rather than the locked-in 10%, B97(a)), and the CSM's 10 at 4%.
    let finance = line(&p, ComponentLine::FinanceExpenses);
    unit("IFE on the PV", finance.0, 197.0);
    unit("CSM accretion", finance.2, 10.0);
    // The discretion adjusts the CSM at the locked-in rate: 61 / 1.1 = 55.
    let future = line(&p, ComponentLine::FutureService);
    unit("discretion, PV", future.0, 55.0);
    unit("discretion, CSM", future.2, -55.0);
    // Current service: RA (10), CSM release 107 = 213 / 394 x 198.
    unit("RA released", line(&p, ComponentLine::RiskExpired).1, -10.0);
    unit("CSM released", line(&p, ComponentLine::CsmRecognised).2, -107.0);
    unit("paid", line(&p, ComponentLine::ClaimsAndExpensesPaid).0, -32.0);
    // Closing: 3,044, 10, 106; a liability of 3,160.
    unit("closing PV", p.closing.present_value, 3_044.0);
    unit("closing RA", p.closing.risk_adjustment, 10.0);
    unit("closing CSM", p.closing.csm, 106.0);
    unit("liability", p.closing.liability(), 3_160.0);
    // The paid death benefit is as revised: no experience adjustment.
    assert!(line(&p, ComponentLine::ExperienceAdjustments).0.abs() < 1e-9);
    // The reconciliation adds up.
    let sum: f64 = p.by_component[..p.by_component.len() - 1]
        .iter()
        .map(|r| r.present_value)
        .sum();
    assert!((sum - p.closing.present_value).abs() < 1e-9);
}

/// A three-year group with premium 1,000 at once and claims of 300 at the
/// middle of each year, recognised at 4%; the current rate moves to 6%,
/// then 2%, then 3%.
fn moving_curve(oci: bool, current: bool) -> Vec<GroupPeriod> {
    let mid = |claims: &[f64]| {
        ExpectedCashFlows::new(claims.len())
            .with_claims(claims.to_vec())
            .with_premiums(vec![0.0; claims.len()])
    };
    let options = GroupOptions {
        oci,
        ..GroupOptions::default()
    };
    let mut first = mid(&[300.0; 3]);
    first.premiums[0] = 1_000.0;
    let mut group = Group::recognise(first, 45.0, flat(0.04), options).unwrap();
    let rates = [0.06, 0.02, 0.03];
    (0..3)
        .map(|k| {
            let actual = paid(if k == 0 { 1_000.0 } else { 0.0 }, 300.0);
            let mut inputs = PeriodInputs::new(actual, mid(&vec![300.0; 2 - k]), 15.0 * (2 - k) as f64)
                .with_risk_adjustment_released(15.0);
            if current {
                inputs = inputs.with_current_curve(flat(rates[k]));
            }
            group.roll_forward(&inputs).unwrap()
        })
        .collect()
}

#[test]
fn the_oci_option_keeps_profit_at_the_locked_in_curve() {
    let locked = moving_curve(false, false);
    let all_in_profit = moving_curve(false, true);
    let with_oci = moving_curve(true, true);
    for k in 0..3 {
        // The current curve changes the liability and the finance expense,
        // not the CSM or the service result.
        let (a, b, c) = (&locked[k], &all_in_profit[k], &with_oci[k]);
        assert!((a.closing.csm - b.closing.csm).abs() < 1e-9);
        assert!((a.insurance_service_result() - b.insurance_service_result()).abs() < 1e-9);
        assert!((b.finance_expenses - c.finance_expenses).abs() < 1e-9);
        // Under the OCI option profit is as if measured at the locked-in
        // curve; OCI takes the rest, and accumulates to the locked-in present
        // value less the current one.
        assert!((c.profit() - a.profit()).abs() < 1e-9, "year {}", k + 1);
        assert!((c.finance_expenses_in_profit() - a.finance_expenses).abs() < 1e-9);
        let accumulated: f64 = with_oci[..=k].iter().map(|p| p.other_comprehensive_income()).sum();
        assert!((accumulated - c.closing.accumulated_oci()).abs() < 1e-9);
        assert_eq!(b.finance_expenses_in_oci, 0.0);
    }
    // In the first year the rate rose: the liability at 6% is smaller, so
    // the finance expense falls and OCI is a gain.
    assert!(with_oci[0].other_comprehensive_income() > 0.0);
    assert!(with_oci[0].closing.present_value < locked[0].closing.present_value);
    // At the end nothing is left to discount: the OCI has recycled to nil.
    assert!(with_oci[2].closing.accumulated_oci().abs() < 1e-9);
    let total_oci: f64 = with_oci.iter().map(|p| p.other_comprehensive_income()).sum();
    assert!(total_oci.abs() < 1e-9);
}

#[test]
fn a_financial_revision_must_match_the_opening_view() {
    let mut group = example_6();
    let inputs = PeriodInputs::new(paid(3_000.0, 31.0), view(&[0.0, 0.0], &[33.0, 3_381.0]), 20.0)
        .with_financial_revision(view(&[0.0], &[31.0]));
    assert!(group.roll_forward(&inputs).is_err());
}
