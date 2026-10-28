//! Mack's assumption tests (Mack 1994, appendices G and H) against R
//! ChainLadder's `cyEffTest` and `dfCorTest` and chainladder-python's
//! `ValuationCorrelation` and `DevelopmentCorrelation` on RAA, GenIns and
//! ABC (`scripts/reserving_assumptions_r.R`,
//! `scripts/reserving_assumptions_python.py`), and a triangle with an
//! injected calendar-year shock, which must fail.
//!
//! Mack's own printed values are on RAA (his example of chapter 6, the RAA
//! data), and are checked in `prospicio-reserving`'s unit tests.

use prospicio_reserving::{
    CalendarYearTest, DevelopmentColumn, DevelopmentCorrelationTest, Grain, Long, Mack, Month,
    Triangle, calendar_year_effect, development_correlation,
};
use prospicio_validation::{Case, check, reference, triangle};

/// The test's value for one reference row; `None` if the row names a
/// diagonal or pair the test does not have.
fn calendar_value(t: &CalendarYearTest, case: &Case) -> Option<f64> {
    let total = match case.get("quantity") {
        "total_z" => Some(t.z),
        "total_expected" => Some(t.expected),
        "total_variance" => Some(t.variance),
        "lower" => Some(t.lower),
        "upper" => Some(t.upper),
        "passes" => Some(f64::from(u8::from(t.passes()))),
        "n_diagonals" => Some(t.diagonals.len() as f64),
        _ => None,
    };
    if total.is_some() {
        return total;
    }
    let year = case.number("arg")? as i32;
    let d = t.diagonals.iter().find(|d| d.valuation.year() == year)?;
    Some(match case.get("quantity") {
        "small" => d.small as f64,
        "large" => d.large as f64,
        "z" => d.z as f64,
        "n" => d.n as f64,
        "m" => d.m as f64,
        "expected" => d.expected,
        "variance" => d.variance,
        _ => return None,
    })
}

fn correlation_value(t: &DevelopmentCorrelationTest, case: &Case) -> Option<f64> {
    Some(match case.get("quantity") {
        "t" => t.t,
        "variance" => t.variance,
        "lower" => t.lower,
        "upper" => t.upper,
        "passes" => f64::from(u8::from(t.passes())),
        "n_pairs" => t.pairs.len() as f64,
        q => {
            let p = t.pairs.get(case.number("arg")? as usize)?;
            match q {
                "t_k" => p.t,
                "weight" => p.weight,
                _ => return None,
            }
        }
    })
}

/// The confidence level of a reference row: `ci=` in its `test`, or the
/// default (chainladder-python's rows are at the defaults).
fn ci(case: &Case, default: f64) -> f64 {
    if case.get("test").contains("ci=") {
        case.param("test", "ci")
    } else {
        default
    }
}

fn eval(case: &Case) -> Option<f64> {
    let tri = triangle(case.get("dataset"));
    let test = case.get("test");
    if test.starts_with("calendar_year_effect") {
        let t = calendar_year_effect(&tri, "values", ci(case, 0.95)).ok()?;
        calendar_value(&t, case)
    } else {
        let t = development_correlation(&tri, "values", ci(case, 0.5)).ok()?;
        correlation_value(&t, case)
    }
}

#[test]
fn matches_r_chainladder() {
    check(&reference("reserving_assumptions_r.csv"), eval);
}

#[test]
fn matches_chainladder_python() {
    check(&reference("reserving_assumptions_python.csv"), eval);
}

#[test]
fn mack_fit_tests_the_same_link_ratios_without_exclusions() {
    for name in ["raa", "genins", "abc"] {
        let tri = triangle(name);
        let fit = Mack::default().fit(&tri, "values").unwrap();
        assert_eq!(
            fit.calendar_year_effect(0.95).unwrap(),
            calendar_year_effect(&tri, "values", 0.95).unwrap()
        );
        assert_eq!(
            fit.development_correlation(0.5).unwrap(),
            development_correlation(&tri, "values", 0.5).unwrap()
        );
    }
}

/// `tri` with the incremental values of the calendar years in `years`
/// multiplied by `factor`: a one-off calendar effect, such as a court
/// ruling or a change in case reserving that lifts every origin's
/// development in those years.
fn shocked(tri: &Triangle, years: &[i32], factor: f64) -> Triangle {
    let [_, _, n_origins, n_dev] = tri.shape();
    let (mut origin, mut age, mut value) = (vec![], vec![], vec![]);
    for o in 0..n_origins {
        let mut cumulative = 0.0;
        let mut previous = 0.0;
        for d in 0..n_dev {
            let Some(c) = tri.get(0, 0, o, d) else {
                continue;
            };
            let first = tri.origins()[o].start();
            let year = first.year() + d as i32;
            let increment = c - previous;
            previous = c;
            cumulative += if years.contains(&year) {
                factor * increment
            } else {
                increment
            };
            origin.push(first);
            age.push(tri.development()[d]);
            value.push(cumulative);
        }
    }
    Triangle::from_long(&Long {
        keys: &[],
        origin: &origin,
        development: DevelopmentColumn::Age(&age),
        values: &[("values", &value)],
        origin_grain: Grain::Year,
        development_grain: Grain::Year,
        cumulative: true,
    })
    .unwrap()
}

/// A cumulative annual triangle of `n` origins from 2001 whose increments
/// follow a fixed payment pattern with up to 5% multiplicative noise from a
/// seeded linear congruential generator: no calendar effect.
fn synthetic(n: usize) -> Triangle {
    let mut state: u64 = 551;
    let mut noise = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        1.0 + 0.1 * ((state >> 11) as f64 / (1u64 << 53) as f64 - 0.5)
    };
    let (mut origin, mut age, mut value) = (vec![], vec![], vec![]);
    for o in 0..n {
        let mut cumulative = 0.0;
        for d in 0..n - o {
            cumulative += 1000.0 * (1.0 + 0.02 * o as f64) * 0.6_f64.powi(d as i32) * noise();
            origin.push(Month::january(2001 + o as i32));
            age.push(12 * (d as u32 + 1));
            value.push(cumulative);
        }
    }
    Triangle::from_long(&Long {
        keys: &[],
        origin: &origin,
        development: DevelopmentColumn::Age(&age),
        values: &[("values", &value)],
        origin_grain: Grain::Year,
        development_grain: Grain::Year,
        cumulative: true,
    })
    .unwrap()
}

#[test]
fn an_injected_calendar_shock_fails_the_test() {
    // Two calendar years whose payments are 50% higher on every origin
    // (2006 and 2010): the link ratios ending in each are all large, so
    // both diagonals have Z_j = 0, and those starting in it are mostly
    // small.
    let tri = synthetic(15);
    let before = calendar_year_effect(&tri, "values", 0.95).unwrap();
    assert!(
        before.passes(),
        "Z = {} in ({}, {})",
        before.z,
        before.lower,
        before.upper
    );
    let years = [2006, 2010];
    let after = calendar_year_effect(&shocked(&tri, &years, 1.5), "values", 0.95).unwrap();
    assert!(
        !after.passes(),
        "Z = {} within ({}, {}) after the shock",
        after.z,
        after.lower,
        after.upper
    );
    for d in &after.diagonals {
        let year = d.valuation.year();
        if years.contains(&year) {
            assert_eq!((d.small, d.z), (0, 0), "{year}");
        }
    }
    // Mack's fit tests the same link ratios.
    let fit = Mack::default()
        .fit(&shocked(&tri, &years, 1.5), "values")
        .unwrap();
    assert!(!fit.calendar_year_effect(0.95).unwrap().passes());
    // On a 10 x 10 triangle the test has little power: RAA and GenIns
    // with the same shock on two diagonals still pass.
    for name in ["raa", "genins"] {
        let tri = triangle(name);
        let first = tri.origins()[0].start().year();
        let t = calendar_year_effect(&shocked(&tri, &[first + 5, first + 7], 1.5), "values", 0.95)
            .unwrap();
        assert!(t.passes(), "{name}");
    }
}
