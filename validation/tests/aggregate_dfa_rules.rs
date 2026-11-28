//! The projection's rules (`docs/design/esg-dfa.md`, PR 4): the AR(2)
//! underwriting cycle's moments against its recursion and its stationary
//! closed form, tax with and without losses carried forward and the
//! capital rule by hand, and the surplus identity with all three.

use prospicio_aggregate::dfa::{CapitalBasis, CapitalRule, Cycle, Line, Projection, Tax};

#[test]
fn the_cycle_has_its_ar2_moments() {
    let (years, n) = (12, 20_000);
    let cycle = Cycle::new(-0.02, 1.2, -0.6, 0.05, 0.08).unwrap();
    let projection = Projection::new(years, 1_000.0, vec![0.0; years * n], n)
        .unwrap()
        .with_premiums(vec![100.0; years])
        .with_cycle(cycle, 7);
    let paths = projection.run_without_economy().unwrap();
    for t in [1, 2, 5, 12] {
        let x: Vec<f64> = (0..n)
            .map(|s| (paths.value(Line::Premium, s, t).unwrap() / 100.0).ln())
            .collect();
        let mean = x.iter().sum::<f64>() / n as f64;
        let var = x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
        let se = (var / n as f64).sqrt();
        assert!((mean - cycle.mean_at(t)).abs() < 4.0 * se, "year {t}: {mean}");
        assert!((var / cycle.variance_at(t) - 1.0).abs() < 0.05, "year {t}: {var}");
    }
    // The recursion reaches the stationary closed form, and an AR(1) is the
    // special case φ2 = 0.
    assert!((cycle.variance_at(400) / cycle.stationary_variance() - 1.0).abs() < 1e-12);
    let ar1 = Cycle::new(0.0, 0.7, 0.0, 0.1, 0.0).unwrap();
    let closed = 0.01 * (1.0 - 0.49f64.powi(6)) / (1.0 - 0.49);
    assert!((ar1.variance_at(6) - closed).abs() < 1e-15);
    assert!(ar1.period().is_none());
}

/// One path whose pre-tax results are -50, 30 and 40: premium 100, no
/// expenses, losses 150, 70 and 60.
fn three_years() -> Projection {
    Projection::new(3, 200.0, vec![150.0, 70.0, 60.0], 1)
        .unwrap()
        .with_premiums(vec![100.0; 3])
}

#[test]
fn tax_carries_losses_forward_or_not() {
    let carried = three_years()
        .with_tax(Tax {
            rate: 0.3,
            carry_forward: true,
        })
        .run_without_economy()
        .unwrap();
    let plain = three_years()
        .with_tax(Tax {
            rate: 0.3,
            carry_forward: false,
        })
        .run_without_economy()
        .unwrap();
    let tax = |p: &prospicio_aggregate::dfa::ProjectionPaths, t| p.value(Line::Tax, 0, t).unwrap();
    // Carried: the 50 lost absorbs year 2's 30 and 20 of year 3's 40.
    for (t, want) in [(1, 0.0), (2, 0.0), (3, 6.0)] {
        assert!((tax(&carried, t) - want).abs() < 1e-12, "carried, year {t}");
    }
    for (t, want) in [(1, 0.0), (2, 9.0), (3, 12.0)] {
        assert!((tax(&plain, t) - want).abs() < 1e-12, "plain, year {t}");
    }
    assert!((carried.value(Line::Surplus, 0, 3).unwrap() - (200.0 + 20.0 - 6.0)).abs() < 1e-12);
}

#[test]
fn the_capital_rule_pays_dividends_and_injects() {
    // Above 1.5 times premium a dividend; below 0.5 times an injection.
    let rule = CapitalRule {
        basis: CapitalBasis::Premium,
        dividend_above: Some(1.5),
        inject_below: Some(0.5),
    };
    let paths = Projection::new(3, 200.0, vec![0.0, 260.0, 60.0], 1)
        .unwrap()
        .with_premiums(vec![100.0; 3])
        .with_capital(rule)
        .run_without_economy()
        .unwrap();
    let v = |line, t| paths.value(line, 0, t).unwrap();
    // Year 1: 200 + 100 = 300, a dividend of 150 leaves 150.
    assert!((v(Line::Dividends, 1) - 150.0).abs() < 1e-12);
    assert!((v(Line::Surplus, 1) - 150.0).abs() < 1e-12);
    // Year 2: 150 - 160 = -10 before capital, an injection of 60 to 50.
    assert!((v(Line::SurplusBeforeCapital, 2) + 10.0).abs() < 1e-12);
    assert!((v(Line::Injections, 2) - 60.0).abs() < 1e-12);
    assert!((v(Line::Surplus, 2) - 50.0).abs() < 1e-12);
    // Ruin reads the surplus before the injection.
    assert_eq!(paths.ruin(3).unwrap().probability, 1.0);
    assert_eq!(paths.ruin(1).unwrap().probability, 0.0);
    // A floor above the dividend threshold is refused.
    let bad = CapitalRule {
        inject_below: Some(2.0),
        ..rule
    };
    assert!(three_years().with_capital(bad).run_without_economy().is_err());
    assert!(
        three_years()
            .with_tax(Tax { rate: 1.5, carry_forward: true })
            .run_without_economy()
            .is_err()
    );
}

#[test]
fn the_surplus_moves_by_the_result_after_tax_and_capital() {
    let (years, n) = (8, 300);
    let losses: Vec<f64> = (0..years * n)
        .map(|i| 60.0 + 40.0 * ((i * 7919 % 101) as f64 / 100.0))
        .collect();
    let paths = Projection::new(years, 150.0, losses, n)
        .unwrap()
        .with_opening_reserves(vec![40.0, 20.0])
        .with_premiums(vec![100.0; years])
        .with_expense_ratio(0.2)
        .with_payout(vec![0.6, 0.4])
        .with_cycle(Cycle::new(0.0, 1.1, -0.5, 0.06, -0.05).unwrap(), 3)
        .with_tax(Tax {
            rate: 0.25,
            carry_forward: true,
        })
        .with_capital(CapitalRule {
            basis: CapitalBasis::Reserves,
            dividend_above: Some(3.0),
            inject_below: Some(0.2),
        })
        .run_without_economy()
        .unwrap();
    for s in 0..n {
        let mut surplus = paths.opening_surplus();
        for t in 1..=years {
            let v = |line| paths.value(line, s, t).unwrap();
            let before = surplus + v(Line::Result) - v(Line::Tax);
            assert!((v(Line::SurplusBeforeCapital) - before).abs() < 1e-9);
            surplus = before - v(Line::Dividends) + v(Line::Injections);
            assert!((v(Line::Surplus) - surplus).abs() < 1e-9, "path {s}, year {t}");
            assert!(v(Line::Surplus) >= 0.2 * v(Line::Reserves) - 1e-9);
            assert!(v(Line::Surplus) <= 3.0 * v(Line::Reserves) + 1e-9);
        }
    }
}
