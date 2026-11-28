//! IFRS 17's premium allocation approach (`ifrs17::paa`), worked by hand:
//! IFRS 17 Example 10 is beyond the text a cloud session could read, so its
//! parity waits (`reserving-ifrs17-measurement.md`, PR 3). The cases cover
//! the LRC and revenue on the pattern, amortised and expensed acquisition
//! cash flows, the LIC with its unwind and changes for past service, the
//! onerous test and its run-off, a revision of the expected premiums, and
//! the reconciliation adding up in each column.

use prospicio_reserving::DiscountCurve;
use prospicio_reserving::ifrs17::CoverageLine;
use prospicio_reserving::ifrs17::paa::{Lic, PaaGroup, PaaInputs, PaaOptions, PaaPeriod};

fn near(name: &str, ours: f64, want: f64) {
    assert!((ours - want).abs() < 1e-9, "{name}: {ours} against {want}");
}

/// Each column of the reconciliation adds up to its closing row.
fn adds_up(p: &PaaPeriod) {
    let rows = &p.by_coverage;
    let (last, body) = rows.split_last().unwrap();
    assert_eq!(last.line, CoverageLine::Closing);
    let sum = |f: fn(&prospicio_reserving::ifrs17::CoverageRow) -> f64| body.iter().map(f).sum::<f64>();
    near("LRC", sum(|r| r.excluding_loss_component), last.excluding_loss_component);
    near("LC", sum(|r| r.loss_component), last.loss_component);
    near("LIC", sum(|r| r.incurred_claims), last.incurred_claims);
}

/// Two years, premium 1,000 at once, acquisition 100 amortised, the LIC at
/// 5%; onerous at the end of year 1.
#[test]
fn a_two_year_group_with_an_onerous_test() {
    let options = PaaOptions {
        lic_curve: Some(DiscountCurve::flat(0.05).unwrap()),
        ..PaaOptions::evenly(2)
    };
    let mut group = PaaGroup::recognise(1_000.0, 100.0, options).unwrap();
    // Year 1: revenue 500 and amortisation 50, so the LRC is
    // 1,000 - 100 + 50 - 500 = 450. Claims of 300 incurred (their
    // fulfilment cash flows), 180 paid, a LIC of 110 + 15: 5 for past
    // service. The remaining coverage's fulfilment cash flows are 600,
    // so a loss component of 150.
    let year_1 = group
        .roll_forward(
            &PaaInputs::new(1_000.0, 100.0, 300.0, 180.0, Lic::new(110.0, 15.0))
                .with_remaining_fulfilment(600.0),
        )
        .unwrap();
    near("revenue", year_1.insurance_revenue, 500.0);
    near("LRC", year_1.closing.lrc_excluding_loss_component, 450.0);
    near("past service", year_1.past_service, 5.0);
    near("loss", year_1.onerous_losses, 150.0);
    near("expenses", year_1.insurance_service_expenses(), 300.0 + 5.0 + 50.0 + 150.0);
    near("finance", year_1.finance_expenses, 0.0);
    near("profit", year_1.profit(), -5.0);
    near("liability", year_1.closing.liability(), 450.0 + 150.0 + 125.0);
    adds_up(&year_1);
    // Year 2: the rest of the revenue and amortisation; the loss component
    // runs off with the coverage; the LIC unwinds 110 x 5% = 5.5; 520
    // incurred, 510 paid, a LIC of 120 + 12: -8.5 for past service.
    let year_2 = group
        .roll_forward(&PaaInputs::new(0.0, 0.0, 520.0, 510.0, Lic::new(120.0, 12.0)))
        .unwrap();
    near("revenue", year_2.insurance_revenue, 500.0);
    near("LRC", year_2.closing.lrc(), 0.0);
    near("reversal", year_2.loss_reversals, 150.0);
    near("finance", year_2.finance_expenses, 5.5);
    near("past service", year_2.past_service, -8.5);
    near("profit", year_2.profit(), 500.0 - (520.0 - 8.5 + 50.0 - 150.0) - 5.5);
    adds_up(&year_2);
    // Over the coverage, profit is cash in less cash out less what is
    // still owed: 1,000 - 100 - 690 - 132.
    near("total", year_1.profit() + year_2.profit(), 1_000.0 - 100.0 - 690.0 - 132.0);
    assert!(group.roll_forward(&PaaInputs::new(0.0, 0.0, 0.0, 0.0, Lic::default())).is_err());
}

/// A year in quarters, acquisition expensed when paid (paragraph 59(a)), the
/// LIC undiscounted, revenue by the expected pattern of claims (B126) and
/// the expected premiums revised.
#[test]
fn quarters_with_expensed_acquisition_and_a_revision() {
    let options = PaaOptions {
        revenue_pattern: vec![1.0, 2.0, 3.0, 4.0],
        expense_acquisition: true,
        lic_curve: None,
        period_years: 0.25,
    };
    let mut group = PaaGroup::recognise(400.0, 40.0, options).unwrap();
    let q1 = group
        .roll_forward(&PaaInputs::new(400.0, 40.0, 30.0, 10.0, Lic::new(20.0, 0.0)))
        .unwrap();
    near("revenue", q1.insurance_revenue, 40.0);
    near("acquisition expensed", q1.acquisition_expense, 40.0);
    near("LRC", q1.closing.lrc(), 360.0);
    adds_up(&q1);
    // The expected premiums rise to 490 (90 more due): the 450 not yet
    // recognised is spread over the remaining weights 2, 3 and 4.
    let q2 = group
        .roll_forward(
            &PaaInputs::new(0.0, 0.0, 70.0, 60.0, Lic::new(30.0, 0.0)).with_expected_premiums(490.0),
        )
        .unwrap();
    near("revenue", q2.insurance_revenue, 450.0 * 2.0 / 9.0);
    near("acquisition", q2.acquisition_expense, 0.0);
    near("past service", q2.past_service, 0.0);
    adds_up(&q2);
    let q3 = group
        .roll_forward(&PaaInputs::new(90.0, 0.0, 100.0, 90.0, Lic::new(40.0, 0.0)))
        .unwrap();
    let q4 = group
        .roll_forward(&PaaInputs::new(0.0, 0.0, 130.0, 170.0, Lic::new(0.0, 0.0)))
        .unwrap();
    let revenue: f64 = [&q1, &q2, &q3, &q4].iter().map(|p| p.insurance_revenue).sum();
    near("revenue in all", revenue, 490.0);
    near("LRC at the end", q4.closing.lrc(), 0.0);
}

#[test]
fn the_inputs_are_checked() {
    assert!(PaaGroup::recognise(1.0, 0.0, PaaOptions::evenly(0)).is_err());
    let bad = PaaOptions {
        revenue_pattern: vec![1.0, -1.0],
        ..PaaOptions::evenly(2)
    };
    assert!(PaaGroup::recognise(1.0, 0.0, bad).is_err());
    let mut group = PaaGroup::recognise(1.0, 0.0, PaaOptions::evenly(1)).unwrap();
    assert!(group.roll_forward(&PaaInputs::new(f64::NAN, 0.0, 0.0, 0.0, Lic::default())).is_err());
    assert!(group.roll_forward(&PaaInputs::new(1.0, 0.0, 0.0, 0.0, Lic::new(0.0, -1.0))).is_err());
}
