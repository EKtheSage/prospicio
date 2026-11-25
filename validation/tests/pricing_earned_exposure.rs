//! Earned exposure from policy dates (`prospicio_pricing::earned`, #981)
//! against a day-by-day count: every policy earns `amount / term` on each
//! day of its term, and a window's figures are sums over its days.

use prospicio_pricing::earned::{Date, EarningMethod, Policy, Windows, earned_exposure};
use prospicio_pricing::ratemaking::PeriodKind;

/// A fixed book: terms of three months to two years, effective on any day
/// from 2019 to 2022, across leap days.
fn book() -> Vec<Policy> {
    let mut state = 20_261_125u64;
    let mut next = |n: u64| {
        state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        (state >> 33) % n
    };
    let first = Date::new(2019, 1, 1).unwrap();
    let day = |offset: i64| {
        // Walk from the first date by whole years, then days.
        let mut date = first;
        let mut left = offset;
        loop {
            let (y, _, _) = date.ymd();
            let next_year = Date::new(y + 1, 1, 1).unwrap();
            let step = next_year.days_since(date);
            if left < step {
                break;
            }
            left -= step;
            date = next_year;
        }
        let (y, _, _) = date.ymd();
        let mut month = 1;
        loop {
            let after = if month == 12 { Date::new(y + 1, 1, 1) } else { Date::new(y, month + 1, 1) }
                .unwrap();
            let start = Date::new(y, month, 1).unwrap();
            let length = after.days_since(start);
            if left < length {
                return Date::new(y, month, left as u8 + 1).unwrap();
            }
            left -= length;
            month += 1;
        }
    };
    (0..60)
        .map(|k| {
            let effective = next(4 * 365) as i64;
            let term = [91, 182, 365, 366, 730][next(5) as usize];
            let amount = 100.0 + next(900) as f64;
            Policy::new(day(effective), day(effective + term), amount)
                .unwrap()
                .with_label(if k % 3 == 0 { "east" } else { "west" })
        })
        .collect()
}

#[test]
fn the_table_matches_a_day_by_day_count() {
    let policies = book();
    let windows = Windows::quarters(2019, 2024).unwrap();
    let bounds = windows.bounds();
    let table = earned_exposure(
        &policies,
        &windows,
        PeriodKind::Calendar,
        EarningMethod::Daily,
        None,
    )
    .unwrap();
    for label in ["east", "west"] {
        let rows = table.label(label);
        assert_eq!(rows.len(), windows.len());
        let mine: Vec<&Policy> = policies.iter().filter(|p| p.label() == label).collect();
        let mut carried = 0.0;
        for (k, row) in rows.iter().enumerate() {
            let (from, to) = (bounds[k], bounds[k + 1]);
            assert_eq!((row.from, row.to), (from, to));
            let (mut written, mut earned, mut unearned, mut in_force) = (0.0, 0.0, 0.0, 0.0);
            for p in &mine {
                let daily = p.amount() / p.term_days() as f64;
                // Days of the term in the window, counted one by one.
                let days_in = (0..to.days_since(from))
                    .filter(|i| {
                        let offset = from.days_since(p.effective()) + i;
                        (0..p.term_days()).contains(&offset)
                    })
                    .count();
                earned += daily * days_in as f64;
                if p.effective() >= from && p.effective() < to {
                    written += p.amount();
                }
                if p.effective() < to {
                    let left = p.expiration().days_since(to).clamp(0, p.term_days());
                    unearned += daily * left as f64;
                    if p.expiration() >= to {
                        in_force += p.amount();
                    }
                }
            }
            assert!((row.written - written).abs() < 1e-9, "{label} window {k}");
            assert!((row.earned - earned).abs() < 1e-9, "{label} window {k}");
            assert!((row.unearned - unearned).abs() < 1e-9, "{label} window {k}");
            assert!((row.in_force - in_force).abs() < 1e-9, "{label} window {k}");
            // Unearned rolls forward by what was written less what was earned.
            carried += row.written - row.earned;
            assert!((row.unearned - carried).abs() < 1e-8, "{label} window {k}");
        }
        // All of it is written and earned inside the windows.
        let total: f64 = mine.iter().map(|p| p.amount()).sum();
        assert!((rows.iter().map(|r| r.written).sum::<f64>() - total).abs() < 1e-9);
        assert!((rows.iter().map(|r| r.earned).sum::<f64>() - total).abs() < 1e-8);
        assert_eq!(rows.last().unwrap().in_force, 0.0);
    }
}

#[test]
fn policy_periods_as_of_a_date_split_what_the_calendar_periods_earned() {
    // As of a valuation date, the policy years' earned amounts and the
    // calendar years' add to the same total, and so do the unearned ones.
    let policies = book();
    let years = Windows::years(2019, 2024).unwrap();
    let valuation = Some(Date::new(2021, 9, 15).unwrap());
    let table = |kind| {
        earned_exposure(&policies, &years, kind, EarningMethod::Daily, valuation).unwrap()
    };
    let (calendar, policy) = (table(PeriodKind::Calendar), table(PeriodKind::Policy));
    let sum = |rows: &[prospicio_pricing::earned::EarnedRow], f: fn(&prospicio_pricing::earned::EarnedRow) -> f64| {
        rows.iter().map(f).sum::<f64>()
    };
    assert!((sum(calendar.rows(), |r| r.earned) - sum(policy.rows(), |r| r.earned)).abs() < 1e-8);
    assert!((sum(calendar.rows(), |r| r.written) - sum(policy.rows(), |r| r.written)).abs() < 1e-9);
    // The calendar table's unearned and in-force figures at the valuation
    // date are in the window it cuts; the policy table spreads them over
    // the policy years.
    for label in ["east", "west"] {
        let cut = &calendar.label(label)[2];
        let spread = policy.label(label);
        assert!((cut.unearned - spread.iter().map(|r| r.unearned).sum::<f64>()).abs() < 1e-8);
        assert!((cut.in_force - spread.iter().map(|r| r.in_force).sum::<f64>()).abs() < 1e-9);
        for row in &spread {
            assert!((row.earned + row.unearned - row.written).abs() < 1e-9);
        }
    }
}
