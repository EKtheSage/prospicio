//! `Trace::first_month` tells a trace from a valuation whatever its balances
//! (#960, #1043): a valuation from month 6 whose trust and operating
//! balances are both exactly 0 is refused by `ActualState::from_trace`, as
//! one with balances is.

use prospicio_pricing::cashflow::{
    ActualState, Base, CashflowTerms, Conditioning, MonthlyPatterns, Requirement, Term,
};

fn terms() -> CashflowTerms {
    let ramp = |m: f64| (0..60).map(|t| ((t + 1) as f64 / m).min(1.0)).collect();
    let patterns = MonthlyPatterns::new(ramp(12.0), ramp(36.0), ramp(48.0)).unwrap();
    let requirement = Requirement::new(vec![Term::new(Base::UnearnedPremium, 0.3)]);
    CashflowTerms::new(100.0, 30, patterns, requirement).unwrap()
}

#[test]
fn a_trace_from_inception_starts_at_month_0() {
    let tr = terms().trace(0.9, 0.2).unwrap();
    assert_eq!(tr.first_month, 0);
    assert!(ActualState::from_trace(&tr, 3).is_ok());
}

#[test]
fn a_valuation_trace_carries_its_month_and_is_refused_even_with_zero_balances() {
    let t = terms();
    let empty = ActualState {
        month: 6,
        funding_reference: Some(0.0),
        ..ActualState::default()
    };
    let tr = t
        .valuation(empty)
        .unwrap()
        .trace(0.9, 0.2, Conditioning::BornhuetterFerguson)
        .unwrap();
    assert_eq!(tr.opening_balance, 0.0);
    assert_eq!(tr.first_month, 6);
    assert!(ActualState::from_trace(&tr, 3).is_err());
}
