//! IFRS 17 reinsurance contracts held (`ifrs17::reinsurance`), worked by
//! hand: IFRS 17 Examples 11 and 12 are beyond the text a cloud session
//! could read, so their parity waits (`reserving-ifrs17-measurement.md`,
//! PR 4). The cases cover the CSM as a net cost and a net gain, its
//! accretion and release, a proportional cover of an onerous group with its
//! loss-recovery component at recognition and after a later loss, the
//! reinsurer's non-performance, a retroactive cover, a current curve, and
//! the reconciliation and the profit or loss adding up each period.

use prospicio_reserving::DiscountCurve;
use prospicio_reserving::ifrs17::reinsurance::{
    CededActual, CededCashFlows, CededLine, CededRow, NonPerformance, ReinsuranceHeld,
    ReinsuranceInputs, ReinsuranceOptions, ReinsurancePeriod,
};
use prospicio_reserving::ifrs17::{ExpectedCashFlows, Group, GroupOptions, PeriodInputs};

fn near(name: &str, ours: f64, want: f64) {
    assert!((ours - want).abs() < 1e-9, "{name}: {ours} against {want}");
}

/// Each column of the reconciliation adds up to its closing row, and the
/// asset moves by the profit or loss and the cash.
fn adds_up(p: &ReinsurancePeriod, opening_asset: f64) {
    let (last, body) = p.by_component.split_last().unwrap();
    assert_eq!(last.line, CededLine::Closing);
    let sum = |f: fn(&CededRow) -> f64| body.iter().map(f).sum::<f64>();
    near("PV", sum(|r| r.present_value), last.present_value);
    near("NPR", sum(|r| r.non_performance), last.non_performance);
    near("RA", sum(|r| r.risk_adjustment), last.risk_adjustment);
    near("CSM", sum(|r| r.csm), last.csm);
    near("asset", last.total(), p.closing.asset());
    let cash = p.component(CededLine::PremiumsPaid).unwrap().present_value
        + p.component(CededLine::RecoveriesReceived).unwrap().present_value;
    near(
        "asset movement",
        p.closing.asset() - opening_asset,
        p.profit() + cash,
    );
    let lr = p.loss_recovery;
    near(
        "loss-recovery component",
        lr.opening + lr.losses - lr.reversals + lr.finance - lr.released,
        lr.closing,
    );
}

fn expected_year(held: &ReinsuranceHeld, ra: f64) -> ReinsuranceInputs {
    let v = held.expected();
    ReinsuranceInputs::new(
        held.as_expected(),
        CededCashFlows::new(v.premiums[1..].to_vec(), v.recoveries[1..].to_vec()),
        ra,
    )
}

/// Premium 300 at once, recoveries of 100 at the end of each of three years,
/// RA 20, 5%: a net cost, accreted and released evenly.
#[test]
fn a_net_cost_accretes_and_is_released_by_coverage_units() {
    let d = |t: f64| 1.05_f64.powf(-t);
    let ceded = CededCashFlows::new(vec![300.0, 0.0, 0.0], vec![100.0; 3]);
    let mut held = ReinsuranceHeld::recognise(
        ceded,
        20.0,
        DiscountCurve::flat(0.05).unwrap(),
        ReinsuranceOptions::default(),
        &[],
    )
    .unwrap();
    let pv0 = -300.0 + 100.0 * (d(1.0) + d(2.0) + d(3.0));
    let csm0 = -(pv0 + 20.0);
    assert!(csm0 > 0.0, "a net cost");
    near("CSM at recognition", held.balances().csm, csm0);
    near("asset at recognition", held.balances().asset(), 0.0);

    let mut opening = 0.0;
    let mut csm = csm0;
    let mut ra = 20.0;
    for year in 1..=3 {
        let ra_close = ra * (3 - year) as f64 / (4 - year) as f64;
        let p = held.roll_forward(&expected_year(&held, ra_close), &[]).unwrap();
        adds_up(&p, opening);
        let accreted = csm * 1.05;
        let released = accreted / (4 - year) as f64;
        near("CSM released", p.allocation.csm_released, released);
        near("closing CSM", p.closing.csm, accreted - released);
        near("RA released", p.allocation.risk_adjustment_released, ra - ra_close);
        near("expected recoveries", p.allocation.expected_recoveries, 100.0);
        near("recovered", p.recovered.total(), 100.0);
        near("finance income on the CSM", p.component(CededLine::FinanceIncome).unwrap().csm, csm * 0.05);
        near("no future service", p.component(CededLine::FutureService).unwrap().total(), 0.0);
        csm = p.closing.csm;
        ra = ra_close;
        opening = p.closing.asset();
    }
    near("nothing left", held.balances().asset(), 0.0);
}

/// Recoveries worth more than the premium: the CSM is a net gain, negative,
/// and its release is income.
#[test]
fn a_net_gain_is_a_negative_csm_released_as_income() {
    let ceded = CededCashFlows::new(vec![200.0, 0.0], vec![120.0; 2]);
    let mut held = ReinsuranceHeld::recognise(
        ceded,
        10.0,
        DiscountCurve::flat(0.0).unwrap(),
        ReinsuranceOptions::default(),
        &[],
    )
    .unwrap();
    near("CSM", held.balances().csm, -50.0);
    let p = held.roll_forward(&expected_year(&held, 5.0), &[]).unwrap();
    adds_up(&p, 0.0);
    near("CSM released", p.allocation.csm_released, -25.0);
    near("net expense", p.net_expense(), 120.0 + 5.0 - 25.0 - 120.0);
}

/// An onerous group (premium 900, claims of 400 a year for three years, RA
/// 30, no discounting: a loss of 330) and a 30% quota share of it.
fn onerous() -> (Group, ReinsuranceHeld) {
    let curve = DiscountCurve::flat(0.0).unwrap();
    let group = Group::recognise(
        ExpectedCashFlows::new(3)
            .with_premiums(vec![900.0, 0.0, 0.0])
            .with_claims(vec![400.0; 3]),
        30.0,
        curve.clone(),
        GroupOptions::default(),
    )
    .unwrap();
    near("underlying loss", group.balances().loss_component, 330.0);
    let held = ReinsuranceHeld::recognise(
        CededCashFlows::new(vec![270.0, 0.0, 0.0], vec![120.0; 3]),
        9.0,
        curve,
        ReinsuranceOptions::default(),
        &[(&group, 0.3)],
    )
    .unwrap();
    (group, held)
}

/// The cover's income at recognition is 30% of the loss, 99; the CSM is
/// adjusted so that the asset carries it as the loss-recovery component,
/// which then runs off with the underlying loss component.
#[test]
fn a_quota_share_of_an_onerous_group_recovers_its_share_of_the_loss() {
    let (mut group, mut held) = onerous();
    let b = held.balances();
    // FCF = -270 + 360 + 9 = 99, the recovery 99: the CSM is nil.
    near("FCF", b.fulfilment_cash_flows(), 99.0);
    near("CSM", b.csm, 0.0);
    near("loss-recovery component", b.loss_recovery, 99.0);
    near("asset", b.asset(), 99.0);

    let mut opening = 0.0;
    for year in 1..=3 {
        let left = 3 - year;
        let u = group
            .roll_forward(&PeriodInputs::new(
                group.as_expected(),
                ExpectedCashFlows::new(left).with_claims(vec![400.0; left]),
                10.0 * left as f64,
            ))
            .unwrap();
        let p = held.roll_forward(&expected_year(&held, 3.0 * left as f64), &[&u]).unwrap();
        adds_up(&p, opening);
        near("tracks the underlying", p.closing.loss_recovery, 0.3 * u.closing.loss_component);
        // The underlying releases 110 of its loss component a year.
        near("released", p.loss_recovery.released, 33.0);
        near("allocation", p.allocation.expected_recoveries, 120.0 - 33.0);
        near("claims recovered", p.recovered.claims, 120.0 - 33.0);
        if year == 1 {
            near("income at recognition", p.recovered.losses, 99.0);
            near("no CSM adjustment after", p.component(CededLine::LossRecovery).unwrap().csm, 0.0);
            // 99 recovered, less the RA of 3 released.
            near("profit", p.profit(), 96.0);
        } else {
            near("no new losses", p.recovered.losses, 0.0);
            near("profit", p.profit(), -3.0);
        }
        near("CSM stays nil", p.closing.csm, 0.0);
        opening = p.closing.asset();
    }
}

/// In year 2 the underlying year-3 claims rise by 100, a further loss; the
/// cover's recoveries rise by 30, which is income, not a CSM adjustment.
#[test]
fn a_later_underlying_loss_is_matched_in_profit_or_loss() {
    let (mut group, mut held) = onerous();
    let u1 = group
        .roll_forward(&PeriodInputs::new(
            group.as_expected(),
            ExpectedCashFlows::new(2).with_claims(vec![400.0; 2]),
            20.0,
        ))
        .unwrap();
    let p1 = held.roll_forward(&expected_year(&held, 6.0), &[&u1]).unwrap();
    let u2 = group
        .roll_forward(&PeriodInputs::new(
            group.as_expected(),
            ExpectedCashFlows::new(1).with_claims(vec![500.0]),
            10.0,
        ))
        .unwrap();
    near("underlying loss", u2.onerous_losses, 100.0);
    let p2 = held
        .roll_forward(
            &ReinsuranceInputs::new(held.as_expected(), CededCashFlows::new(vec![0.0], vec![150.0]), 3.0),
            &[&u2],
        )
        .unwrap();
    adds_up(&p2, p1.closing.asset());
    near("income", p2.recovered.losses, 30.0);
    near("future service", p2.component(CededLine::FutureService).unwrap().csm, -30.0);
    near("matched", p2.component(CededLine::LossRecovery).unwrap().csm, 30.0);
    near("CSM", p2.closing.csm, 0.0);
    near("tracks the underlying", p2.closing.loss_recovery, 0.3 * u2.closing.loss_component);
}

/// A 2% annual default probability with half the recovery lost: the
/// reduction raises the net cost at recognition, and its later change is in
/// profit or loss, leaving the CSM's run-off as it was.
#[test]
fn non_performance_reduces_the_recoveries_and_its_change_is_not_csm() {
    let np = NonPerformance::new(0.02, 0.5).unwrap();
    let ceded = || CededCashFlows::new(vec![300.0, 0.0, 0.0], vec![100.0; 3]);
    let curve = DiscountCurve::flat(0.0).unwrap();
    let options = ReinsuranceOptions {
        non_performance: Some(np),
        ..Default::default()
    };
    let mut held = ReinsuranceHeld::recognise(ceded(), 15.0, curve.clone(), options, &[]).unwrap();
    let npr0 = 50.0 * (0.02 + (1.0 - 0.98_f64.powi(2)) + (1.0 - 0.98_f64.powi(3)));
    near("NPR", held.balances().non_performance, npr0);
    near("CSM", held.balances().csm, npr0 - 15.0);

    let revised = NonPerformance::new(0.05, 0.5).unwrap();
    let p = held
        .roll_forward(&expected_year(&held, 10.0).with_non_performance(revised), &[])
        .unwrap();
    adds_up(&p, 0.0);
    let npr1 = 50.0 * (0.05 + (1.0 - 0.95_f64.powi(2)));
    near("closing NPR", p.closing.non_performance, npr1);
    near("in profit or loss", p.recovered.non_performance, npr0 - npr1);
    near("CSM untouched", p.closing.csm, (npr0 - 15.0) * 2.0 / 3.0);
    // The recoveries received in full: experience of the year-1 reduction.
    near("experience", p.component(CededLine::ExperienceAdjustments).unwrap().total(), 0.0);
}

/// A cover of past events at a net cost of 40 expenses it at once.
#[test]
fn a_retroactive_net_cost_is_expensed_at_recognition() {
    let options = ReinsuranceOptions {
        retroactive: true,
        ..Default::default()
    };
    let mut held = ReinsuranceHeld::recognise(
        CededCashFlows::new(vec![250.0, 0.0], vec![100.0; 2]),
        10.0,
        DiscountCurve::flat(0.0).unwrap(),
        options,
        &[],
    )
    .unwrap();
    near("CSM", held.balances().csm, 0.0);
    near("asset", held.balances().asset(), -40.0);
    let p = held.roll_forward(&expected_year(&held, 5.0), &[]).unwrap();
    adds_up(&p, 0.0);
    near("past events", p.past_events_cost, 40.0);
    near("net expense", p.net_expense(), 40.0 + 5.0);
}

/// A current curve at the period end moves the present value through
/// finance income, not the CSM.
#[test]
fn a_current_curve_is_a_finance_effect() {
    let make = || {
        ReinsuranceHeld::recognise(
            CededCashFlows::new(vec![300.0, 0.0, 0.0], vec![110.0; 3]),
            12.0,
            DiscountCurve::flat(0.03).unwrap(),
            ReinsuranceOptions::default(),
            &[],
        )
        .unwrap()
    };
    let (mut a, mut b) = (make(), make());
    let pa = a.roll_forward(&expected_year(&a, 8.0), &[]).unwrap();
    let pb = b
        .roll_forward(
            &expected_year(&b, 8.0).with_current_curve(DiscountCurve::flat(0.06).unwrap()),
            &[],
        )
        .unwrap();
    adds_up(&pb, 0.0);
    near("CSM", pb.closing.csm, pa.closing.csm);
    near("locked-in PV", pb.closing.present_value_locked_in, pa.closing.present_value);
    let pv_current = 110.0 / 1.06 + 110.0 / 1.06_f64.powi(2);
    near("current PV", pb.closing.present_value, pv_current);
    near(
        "finance",
        pb.finance_income - pa.finance_income,
        pv_current - pa.closing.present_value,
    );
}

#[test]
fn bad_inputs_are_refused() {
    let curve = DiscountCurve::flat(0.0).unwrap();
    let ceded = || CededCashFlows::new(vec![100.0, 0.0], vec![60.0; 2]);
    let opts = ReinsuranceOptions::default;
    assert!(
        ReinsuranceHeld::recognise(CededCashFlows::new(vec![1.0], vec![1.0; 2]), 0.0, curve.clone(), opts(), &[])
            .is_err()
    );
    assert!(ReinsuranceHeld::recognise(ceded(), -1.0, curve.clone(), opts(), &[]).is_err());
    assert!(NonPerformance::new(1.0, 0.5).is_err());
    assert!(NonPerformance::new(0.1, 1.5).is_err());
    let (group, _) = onerous();
    assert!(ReinsuranceHeld::recognise(ceded(), 0.0, curve.clone(), opts(), &[(&group, 1.2)]).is_err());
    let mut held = ReinsuranceHeld::recognise(ceded(), 0.0, curve, opts(), &[(&group, 0.5)]).unwrap();
    let inputs = ReinsuranceInputs::new(CededActual::default(), CededCashFlows::new(vec![0.0], vec![60.0]), 0.0);
    assert!(held.roll_forward(&inputs, &[]).is_err(), "one underlying period is needed");
}
