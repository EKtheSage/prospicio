use super::*;

fn curve() -> DiscountCurve {
    DiscountCurve::flat(0.05).unwrap()
}

fn example_1(claims: f64) -> Group {
    let cash = ExpectedCashFlows::new(3)
        .with_premiums(vec![900.0, 0.0, 0.0])
        .with_claims(vec![claims; 3]);
    Group::recognise(cash, 120.0, curve(), GroupOptions::default()).unwrap()
}

/// Every row list runs from its opening to its closing, and both
/// reconciliations close on the same liability.
fn assert_closes(p: &GroupPeriod) {
    let sum: f64 = p.by_component[..p.by_component.len() - 1]
        .iter()
        .map(ComponentRow::total)
        .sum();
    let closing = p.component(ComponentLine::Closing).unwrap().total();
    assert!((sum - closing).abs() < 1e-9, "{sum} vs {closing}");
    let sum: f64 = p.by_coverage[..p.by_coverage.len() - 1]
        .iter()
        .map(CoverageRow::total)
        .sum();
    let closing_cov = p.coverage(CoverageLine::Closing).unwrap();
    assert!(
        (sum - closing_cov.total()).abs() < 1e-9,
        "{sum} vs {}",
        closing_cov.total()
    );
    assert!((closing - closing_cov.total()).abs() < 1e-9);
    for (columns, closing) in [
        (
            p.by_coverage
                .iter()
                .map(|r| r.loss_component)
                .collect::<Vec<_>>(),
            closing_cov.loss_component,
        ),
        (
            p.by_coverage.iter().map(|r| r.incurred_claims).collect(),
            closing_cov.incurred_claims,
        ),
    ] {
        let sum: f64 = columns[..columns.len() - 1].iter().sum();
        assert!((sum - closing).abs() < 1e-9, "{sum} vs {closing}");
    }
}

#[test]
fn a_profitable_group_has_a_csm_and_an_onerous_one_a_loss_component() {
    let profitable = example_1(200.0).balances();
    assert!(profitable.csm > 0.0 && profitable.loss_component == 0.0);
    let onerous = example_1(400.0).balances();
    assert!(onerous.csm == 0.0 && onerous.loss_component > 0.0);
    assert!((onerous.loss_component - onerous.fulfilment_cash_flows()).abs() < 1e-12);
}

#[test]
fn both_reconciliations_close_with_surprises_every_period() {
    let mut g = Group::recognise(
        ExpectedCashFlows::new(3)
            .with_premiums(vec![500.0, 300.0, 200.0])
            .with_claims(vec![300.0, 300.0, 300.0])
            .with_expenses(vec![20.0; 3])
            .with_acquisition(vec![50.0, 0.0, 0.0])
            .with_investment_component(vec![30.0; 3])
            .with_timing(
                CashFlowTiming::StartOfPeriod,
                CashFlowTiming::MidPeriod,
                CashFlowTiming::StartOfPeriod,
            ),
        60.0,
        DiscountCurve::spot(&[1.0, 3.0], &[0.03, 0.05]).unwrap(),
        GroupOptions {
            coverage_units: Some(vec![3.0, 2.0, 1.0]),
            discount_coverage_units: true,
            period_years: 1.0,
            ..GroupOptions::default()
        },
    )
    .unwrap();
    let periods = [
        (
            520.0,
            5.0,
            340.0,
            55.0,
            vec![290.0, 260.0],
            vec![360.0, 380.0],
            45.0,
        ),
        (280.0, 0.0, 250.0, 0.0, vec![200.0], vec![500.0], 30.0),
        (200.0, 0.0, 600.0, 0.0, vec![], vec![], 0.0),
    ];
    for (premiums, future, claims, acquisition, p_next, c_next, ra) in periods {
        let n = p_next.len();
        let inputs = PeriodInputs::new(
            ActualCashFlows {
                premiums,
                premiums_for_future_service: future,
                claims,
                expenses: 25.0,
                acquisition,
                investment_component: 30.0,
            },
            ExpectedCashFlows::new(n)
                .with_premiums(p_next)
                .with_claims(c_next)
                .with_expenses(vec![20.0; n])
                .with_investment_component(vec![30.0; n])
                .with_timing(
                    CashFlowTiming::StartOfPeriod,
                    CashFlowTiming::MidPeriod,
                    CashFlowTiming::StartOfPeriod,
                ),
            ra,
        );
        let p = g.roll_forward(&inputs).unwrap();
        assert_closes(&p);
        assert!(p.closing.csm >= 0.0 && p.closing.loss_component >= 0.0);
        assert!(p.closing.csm == 0.0 || p.closing.loss_component == 0.0);
    }
    let last = g.balances();
    assert!(last.liability().abs() < 1e-9 && last.loss_component == 0.0);
    assert!(
        g.roll_forward(&PeriodInputs::new(
            ActualCashFlows::default(),
            ExpectedCashFlows::new(0),
            0.0
        ))
        .is_err()
    );
}

#[test]
fn a_group_without_surprises_releases_everything_by_the_end() {
    let mut g = example_1(200.0);
    let mut profit = 0.0;
    for left in [2, 1, 0] {
        let p = g
            .roll_forward(&PeriodInputs::new(
                g.as_expected(),
                ExpectedCashFlows::new(left).with_claims(vec![200.0; left]),
                40.0 * left as f64,
            ))
            .unwrap();
        assert_closes(&p);
        profit += p.profit();
    }
    // The total profit is the premium less the claims paid.
    assert!((profit - 300.0).abs() < 1e-9, "{profit}");
    assert!(g.balances().liability().abs() < 1e-9);
}

#[test]
fn the_inputs_are_checked() {
    let cash = ExpectedCashFlows::new(2).with_claims(vec![1.0]);
    assert!(matches!(
        Group::recognise(cash, 0.0, curve(), GroupOptions::default()),
        Err(Error::LengthMismatch { .. })
    ));
    let cash = ExpectedCashFlows::new(1).with_claims(vec![f64::NAN]);
    assert!(Group::recognise(cash, 0.0, curve(), GroupOptions::default()).is_err());
    let cash = ExpectedCashFlows::new(1);
    assert!(Group::recognise(cash.clone(), -1.0, curve(), GroupOptions::default()).is_err());
    let none = GroupOptions {
        coverage_units: Some(vec![0.0]),
        ..GroupOptions::default()
    };
    assert!(Group::recognise(cash.clone(), 0.0, curve(), none).is_err());
    let mut g = Group::recognise(cash, 0.0, curve(), GroupOptions::default()).unwrap();
    let bad = PeriodInputs::new(
        ActualCashFlows::default(),
        ExpectedCashFlows::new(0),
        f64::INFINITY,
    );
    assert!(g.roll_forward(&bad).is_err());
}
