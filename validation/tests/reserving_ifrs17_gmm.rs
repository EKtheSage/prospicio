//! IFRS 17 general measurement model against the IASB's Illustrative
//! Examples (IFRS Foundation, IFRS 17 Illustrative Examples, 2025 edition):
//! Example 1 (initial recognition), 2A and 2B (subsequent measurement, a
//! favourable change and an onerous one), 3A (insurance revenue with an
//! investment component), 7 (acquisition cash flows) and 8 (the loss
//! component's allocation and reversal).
//!
//! The examples round to whole currency units at each step and chain on the
//! rounded values, so every figure is matched to the unit
//! (`knowledge/references/ifrs17-illustrative-examples.md`).

use prospicio_reserving::DiscountCurve;
use prospicio_reserving::ifrs17::{
    ActualCashFlows, ComponentLine, CoverageLine, ExpectedCashFlows, Group, GroupOptions,
    GroupPeriod, PeriodInputs,
};

/// The examples' figures are rounded at each step: one unit either way.
fn unit(name: &str, ours: f64, printed: f64) {
    assert!(
        (ours - printed).abs() <= 1.0,
        "{name}: {ours:.2} against the printed {printed}"
    );
}

/// Example 8's later years chain on figures rounded twice (the closing
/// present value of 94 is 743 + 37 - 286 - 400, against 100 / 1.05 = 95.24
/// exactly), so they are matched within one and a half units.
fn chained(name: &str, ours: f64, printed: f64) {
    assert!(
        (ours - printed).abs() <= 1.5,
        "{name}: {ours:.2} against the printed {printed}"
    );
}

fn five_percent() -> DiscountCurve {
    DiscountCurve::flat(0.05).unwrap()
}

fn claims(amounts: &[f64]) -> ExpectedCashFlows {
    ExpectedCashFlows::new(amounts.len()).with_claims(amounts.to_vec())
}

/// Example 1's fact pattern: premium 900 at once, claims at the end of each
/// of three years, RA 120 released evenly, 5%.
fn example_1(claim: f64, investment: f64) -> Group {
    let cash = claims(&[claim; 3])
        .with_premiums(vec![900.0, 0.0, 0.0])
        .with_investment_component(vec![investment; 3]);
    Group::recognise(cash, 120.0, five_percent(), GroupOptions::default()).unwrap()
}

fn paid(claims: f64, investment_component: f64) -> ActualCashFlows {
    ActualCashFlows {
        claims,
        investment_component,
        ..ActualCashFlows::default()
    }
}

fn line(p: &GroupPeriod, line: ComponentLine) -> (f64, f64, f64) {
    let r = p.component(line).unwrap();
    (r.present_value, r.risk_adjustment, r.csm)
}

fn current_service(p: &GroupPeriod) -> f64 {
    [
        ComponentLine::CsmRecognised,
        ComponentLine::RiskExpired,
        ComponentLine::ExperienceAdjustments,
    ]
    .iter()
    .map(|&l| p.component(l).unwrap().total())
    .sum()
}

#[test]
fn example_1_initial_recognition() {
    // IE5 to IE11: 1A a CSM of 235; 1B a loss of 309 and no CSM.
    let a = example_1(200.0, 0.0).balances();
    unit("1A present value of outflows", a.present_value + 900.0, 545.0);
    unit("1A fulfilment cash flows", a.fulfilment_cash_flows(), -235.0);
    unit("1A CSM", a.csm, 235.0);
    assert_eq!(a.loss_component, 0.0);
    let b = example_1(400.0, 0.0).balances();
    unit("1B present value of outflows", b.present_value + 900.0, 1089.0);
    unit("1B fulfilment cash flows", b.fulfilment_cash_flows(), 309.0);
    assert_eq!(b.csm, 0.0);
    unit("1B loss component", b.loss_component, 309.0);
}

/// Example 2's year 1, all as expected (IE16, IE17).
fn example_2_year_1(group: &mut Group, investment: f64) -> GroupPeriod {
    let view = claims(&[200.0, 200.0]).with_investment_component(vec![investment; 2]);
    group
        .roll_forward(&PeriodInputs::new(group.as_expected(), view, 80.0))
        .unwrap()
}

#[test]
fn example_2a_a_favourable_change_for_future_service() {
    let mut g = example_1(200.0, 0.0);

    // IE17: year 1.
    let y1 = example_2_year_1(&mut g, 0.0);
    let (pv, ra, csm) = line(&y1, ComponentLine::NewContracts);
    unit("new contracts PV", pv, -355.0);
    unit("new contracts RA", ra, 120.0);
    unit("new contracts CSM", csm, 235.0);
    let (pv, _, csm) = line(&y1, ComponentLine::FinanceExpenses);
    unit("year 1 finance on the PV", pv, 27.0);
    unit("year 1 CSM accretion", csm, 12.0);
    unit("year 1 RA released", line(&y1, ComponentLine::RiskExpired).1, -40.0);
    unit("year 1 CSM released", line(&y1, ComponentLine::CsmRecognised).2, -82.0);
    unit("year 1 current service", current_service(&y1), -122.0);
    unit("year 1 closing PV", y1.closing.present_value, 372.0);
    unit("year 1 closing CSM", y1.closing.csm, 165.0);
    unit("year 1 liability", y1.closing.liability(), 617.0);

    // IE18 to IE20: claims of 150 instead of 200; year 3's revised to 140, RA 30.
    let y2 = g
        .roll_forward(&PeriodInputs::new(paid(150.0, 0.0), claims(&[140.0]), 30.0))
        .unwrap();
    let (pv, _, csm) = line(&y2, ComponentLine::FinanceExpenses);
    unit("year 2 finance on the PV", pv, 19.0);
    unit("year 2 CSM accretion", csm, 8.0);
    let (pv, ra, csm) = line(&y2, ComponentLine::FutureService);
    unit("year 2 future service PV", pv, -58.0);
    unit("year 2 future service RA", ra, -10.0);
    unit("year 2 future service CSM", csm, 68.0);
    unit("year 2 experience", line(&y2, ComponentLine::ExperienceAdjustments).0, -50.0);
    unit("year 2 RA released", line(&y2, ComponentLine::RiskExpired).1, -40.0);
    unit("year 2 CSM released", line(&y2, ComponentLine::CsmRecognised).2, -121.0);
    unit("year 2 current service", current_service(&y2), -211.0);
    unit("year 2 closing PV", y2.closing.present_value, 133.0);
    unit("year 2 closing CSM", y2.closing.csm, 120.0);
    unit("year 2 liability", y2.closing.liability(), 283.0);

    // IE21, IE22: year 3 as revised.
    let y3 = g
        .roll_forward(&PeriodInputs::new(paid(140.0, 0.0), claims(&[]), 0.0))
        .unwrap();
    let (pv, _, csm) = line(&y3, ComponentLine::FinanceExpenses);
    unit("year 3 finance on the PV", pv, 7.0);
    unit("year 3 CSM accretion", csm, 6.0);
    unit("year 3 CSM released", line(&y3, ComponentLine::CsmRecognised).2, -126.0);
    unit("year 3 current service", current_service(&y3), -156.0);
    assert!(y3.closing.liability().abs() < 1e-9);

    // IE23: profit 83, 184 and 143; finance expenses 39, 27 and 13.
    for (p, profit, finance) in [(&y1, 83.0, 39.0), (&y2, 184.0, 27.0), (&y3, 143.0, 13.0)] {
        unit("profit", p.profit(), profit);
        unit("finance expenses", p.finance_expenses, finance);
        unit("service result", p.insurance_service_result(), profit + finance);
        assert_eq!(p.onerous_losses, 0.0);
    }
}

#[test]
fn example_2b_an_unfavourable_change_makes_the_group_onerous() {
    let mut g = example_1(200.0, 0.0);
    example_2_year_1(&mut g, 0.0);

    // IE24 to IE26: claims of 400; year 3's revised to 450, RA to 88.
    let y2 = g
        .roll_forward(&PeriodInputs::new(paid(400.0, 0.0), claims(&[450.0]), 88.0))
        .unwrap();
    let (pv, ra, csm) = line(&y2, ComponentLine::FutureService);
    unit("year 2 future service PV", pv, 238.0);
    unit("year 2 future service RA", ra, 48.0);
    unit("year 2 future service CSM", csm, -173.0);
    unit("year 2 future service total", pv + ra + csm, 113.0);
    unit("year 2 experience", line(&y2, ComponentLine::ExperienceAdjustments).0, 200.0);
    assert_eq!(line(&y2, ComponentLine::CsmRecognised).2, 0.0);
    unit("year 2 current service", current_service(&y2), 160.0);
    unit("year 2 onerous loss", y2.onerous_losses, 113.0);
    unit("year 2 loss component", y2.closing.loss_component, 113.0);
    unit("year 2 closing PV", y2.closing.present_value, 429.0);
    assert_eq!(y2.closing.csm, 0.0);
    unit("year 2 liability", y2.closing.liability(), 517.0);

    // IE27: year 3 as revised; the loss component runs off with it.
    let y3 = g
        .roll_forward(&PeriodInputs::new(paid(450.0, 0.0), claims(&[]), 0.0))
        .unwrap();
    unit("year 3 finance", y3.finance_expenses, 21.0);
    unit("year 3 current service", current_service(&y3), -88.0);
    assert!(y3.closing.liability().abs() < 1e-9);
    assert!(y3.closing.loss_component.abs() < 1e-9);

    // IE28: profit 83, (300) and 67; the loss of 113 in year 2.
    unit("year 2 profit", y2.profit(), -300.0);
    unit("year 3 profit", y3.profit(), 67.0);
    unit("year 3 service result", y3.insurance_service_result(), 88.0);
}

#[test]
fn example_3a_insurance_revenue_excludes_the_investment_component() {
    // IE30 to IE36: Example 2A with an investment component of 100 a year.
    let mut g = example_1(200.0, 100.0);
    let y1 = example_2_year_1(&mut g, 100.0);
    let y2 = g
        .roll_forward(&PeriodInputs::new(
            paid(150.0, 100.0),
            claims(&[140.0]).with_investment_component(vec![100.0]),
            30.0,
        ))
        .unwrap();
    let y3 = g
        .roll_forward(&PeriodInputs::new(paid(140.0, 100.0), claims(&[]), 0.0))
        .unwrap();
    // Revenue 222 = 100 + 40 + 82; 261 = 100 + 40 + 121 (50 incurred and
    // the experience adjustment of 50); 196 = 40 + 30 + 126.
    for (p, revenue, expenses, expected, ra, csm) in [
        (&y1, 222.0, 100.0, 100.0, 40.0, 82.0),
        (&y2, 261.0, 50.0, 100.0, 40.0, 121.0),
        (&y3, 196.0, 40.0, 40.0, 30.0, 126.0),
    ] {
        unit("revenue", p.insurance_revenue(), revenue);
        unit("service expenses", p.insurance_service_expenses(), expenses);
        unit("expected claims released", p.revenue.expected_claims_and_expenses, expected);
        unit("RA released", p.revenue.risk_adjustment_released, ra);
        unit("CSM released", p.revenue.csm_released, csm);
        let ic = p.coverage(CoverageLine::InvestmentComponents).unwrap();
        assert_eq!((ic.excluding_loss_component, ic.incurred_claims), (-100.0, 100.0));
    }
    // IE33: the liability at the end of year 1 is 617, as in Example 2.
    unit("year 1 liability", y1.closing.liability(), 617.0);
    // The profit is Example 2A's: an investment component changes the
    // presentation, not the result.
    unit("year 2 profit", y2.profit(), 184.0);
}

#[test]
fn example_7_acquisition_cash_flows() {
    // IE74 to IE80: premium 900 at once, claims of 200 a year, acquisition
    // cash flows of 90 at the start, RA 15 released evenly, no discounting.
    let cash = claims(&[200.0; 3])
        .with_premiums(vec![900.0, 0.0, 0.0])
        .with_acquisition(vec![90.0, 0.0, 0.0]);
    let flat = DiscountCurve::flat(0.0).unwrap();
    let mut g = Group::recognise(cash, 15.0, flat, GroupOptions::default()).unwrap();
    unit("fulfilment cash flows", g.balances().fulfilment_cash_flows(), -195.0);
    unit("CSM", g.balances().csm, 195.0);
    let closing = [(410.0, 540.0), (205.0, 270.0), (0.0, 0.0)];
    for (k, (fcf, liability)) in closing.into_iter().enumerate() {
        let left = 2 - k;
        let p = g
            .roll_forward(&PeriodInputs::new(
                g.as_expected(),
                claims(&vec![200.0; left]),
                5.0 * left as f64,
            ))
            .unwrap();
        // IE78 to IE80: revenue 300 = 200 + 5 + 65 + 30 and expenses
        // 230 = 200 + 30, every year.
        unit("CSM released", p.revenue.csm_released, 65.0);
        unit("acquisition recovered", p.revenue.acquisition_recovered, 30.0);
        unit("revenue", p.insurance_revenue(), 300.0);
        unit("service expenses", p.insurance_service_expenses(), 230.0);
        unit("service result", p.insurance_service_result(), 70.0);
        unit("fulfilment cash flows", p.closing.fulfilment_cash_flows(), fcf);
        unit("liability", p.closing.liability(), liability);
    }
}

#[test]
fn example_8_the_loss_component_is_allocated_and_reversed() {
    // IE82 to IE90: premium 800 at once, claims of 400 a year, RA 240
    // released evenly, 5%; year 3's claims revised to 100 at the end of
    // year 2.
    let cash = claims(&[400.0; 3]).with_premiums(vec![800.0, 0.0, 0.0]);
    let mut g = Group::recognise(cash, 240.0, five_percent(), GroupOptions::default()).unwrap();
    unit("loss at recognition", g.balances().loss_component, 529.0);

    // IE91 to IE93: year 1.
    let y1 = g
        .roll_forward(&PeriodInputs::new(g.as_expected(), claims(&[400.0, 400.0]), 160.0))
        .unwrap();
    unit("year 1 share", 1000.0 * y1.loss_component_share, 398.0);
    unit("year 1 loss", y1.onerous_losses, 529.0);
    let fin = y1.coverage(CoverageLine::FinanceExpenses).unwrap();
    unit("year 1 finance, rest of the LRC", fin.excluding_loss_component, 33.0);
    unit("year 1 finance, loss component", fin.loss_component, 21.0);
    unit("year 1 finance", y1.finance_expenses, 54.0);
    let incurred = y1.coverage(CoverageLine::IncurredClaims).unwrap();
    unit("year 1 loss component released", incurred.loss_component, -191.0);
    unit("year 1 revenue", y1.insurance_revenue(), 289.0);
    unit("year 1 incurred claims", y1.incurred_claims, 209.0);
    unit("year 1 loss component", y1.closing.loss_component, 359.0);
    unit("year 1 rest of the LRC", y1.closing.excluding_loss_component(), 544.0);
    unit("year 1 liability", y1.closing.liability(), 903.0);
    unit("year 1 closing PV", y1.closing.present_value, 743.0);

    // IE94 to IE96: year 2, year 3's claims revised to 100.
    let y2 = g
        .roll_forward(&PeriodInputs::new(paid(400.0, 0.0), claims(&[100.0]), 80.0))
        .unwrap();
    chained("year 2 share", 1000.0 * y2.loss_component_share, 398.0);
    let fin = y2.coverage(CoverageLine::FinanceExpenses).unwrap();
    chained("year 2 finance, rest of the LRC", fin.excluding_loss_component, 22.0);
    chained("year 2 finance, loss component", fin.loss_component, 15.0);
    let (pv, _, csm) = line(&y2, ComponentLine::FutureService);
    chained("year 2 future service PV", pv, -286.0);
    chained("year 2 future service CSM", csm, 103.0);
    chained("year 2 loss reversed", y2.loss_reversals, 183.0);
    chained("year 2 CSM released", y2.revenue.csm_released, 52.0);
    chained("year 2 revenue", y2.insurance_revenue(), 341.0);
    chained("year 2 incurred claims", y2.incurred_claims, 209.0);
    assert_eq!(y2.closing.loss_component, 0.0);
    chained("year 2 closing PV", y2.closing.present_value, 94.0);
    chained("year 2 closing CSM", y2.closing.csm, 51.0);
    chained("year 2 liability", y2.closing.liability(), 225.0);

    // IE97, IE98: year 3 as revised (the printed figures carry a rounding
    // difference of 1).
    let y3 = g
        .roll_forward(&PeriodInputs::new(paid(100.0, 0.0), claims(&[]), 0.0))
        .unwrap();
    chained("year 3 finance", y3.finance_expenses, 8.0);
    chained("year 3 CSM released", y3.revenue.csm_released, 54.0);
    chained("year 3 revenue", y3.insurance_revenue(), 233.0);
    chained("year 3 service expenses", y3.insurance_service_expenses(), 100.0);
    assert!(y3.closing.liability().abs() < 1e-9);
}
