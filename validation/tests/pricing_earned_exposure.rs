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

/// Werner and Modlin's six policies of chapter 4 (*Basic Ratemaking*, 5th
/// ed., Tables 4.7 to 4.15): one exposure each, effective a quarter apart
/// from 1 October 2010, for `months` months. The book writes the
/// expiration as the last day in force; here it is the day after.
fn werner_modlin(months: u8, amount: f64) -> Vec<Policy> {
    [(2010, 10), (2011, 1), (2011, 4), (2011, 7), (2011, 10), (2012, 1)]
        .into_iter()
        .map(|(year, month)| {
            let end = month + months;
            let expiration = if end > 12 {
                Date::new(year + 1, end - 12, 1)
            } else {
                Date::new(year, end, 1)
            }
            .unwrap();
            Policy::new(Date::new(year, month, 1).unwrap(), expiration, amount).unwrap()
        })
        .collect()
}

fn by_year(policies: &[Policy], kind: PeriodKind, valuation: Option<Date>) -> Vec<[f64; 2]> {
    earned_exposure(
        policies,
        &Windows::years(2010, 2012).unwrap(),
        kind,
        EarningMethod::Daily,
        valuation,
    )
    .unwrap()
    .rows()
    .iter()
    .map(|r| [r.written, r.earned])
    .collect()
}

#[test]
fn werner_modlin_chapter_4_annual_policies() {
    let policies = werner_modlin(12, 1.0);
    // As of 12/31/12, which is a valuation date of 1 January 2013.
    let valuation = Some(Date::new(2013, 1, 1).unwrap());

    // Table 4.7, calendar year written exposures, and Table 4.11, earned.
    // The book counts months (a quarter of a policy is 0.25); exact days
    // give 92/365 for the same quarter, within two days in the year.
    let calendar = by_year(&policies, PeriodKind::Calendar, valuation);
    for (row, (written, earned)) in calendar.iter().zip([(1.0, 0.25), (4.0, 3.25), (1.0, 2.50)]) {
        assert_eq!(row[0], written);
        assert!((row[1] - earned).abs() < 6e-3, "{} against {earned}", row[1]);
    }
    // Tables 4.9 and 4.13, policy year written and earned exposures: all
    // six policies have expired by the valuation date.
    let policy = by_year(&policies, PeriodKind::Policy, valuation);
    for (row, written) in policy.iter().zip([1.0, 4.0, 1.0]) {
        assert_eq!(row[0], written);
        assert!((row[1] - written).abs() < 1e-12);
    }
    // Table 4.15, in-force exposure as of 01/01/11, 06/15/11 and 01/01/12.
    let on = |y, m, d| prospicio_pricing::earned::in_force(&policies, Date::new(y, m, d).unwrap());
    assert_eq!([on(2011, 1, 1), on(2011, 6, 15), on(2012, 1, 1)], [2.0, 3.0, 4.0]);
}

#[test]
fn werner_modlin_chapter_4_six_month_policies() {
    // Tables 4.18 to 4.21: each six-month policy is half an exposure.
    let policies = werner_modlin(6, 0.5);
    let valuation = Some(Date::new(2013, 1, 1).unwrap());
    let calendar = by_year(&policies, PeriodKind::Calendar, valuation);
    for (row, (written, earned)) in calendar.iter().zip([(0.5, 0.25), (2.0, 2.00), (0.5, 0.75)]) {
        assert_eq!(row[0], written);
        assert!((row[1] - earned).abs() < 6e-3, "{} against {earned}", row[1]);
    }
    let policy = by_year(&policies, PeriodKind::Policy, valuation);
    for (row, written) in policy.iter().zip([0.5, 2.0, 0.5]) {
        assert_eq!(row[0], written);
        assert!((row[1] - written).abs() < 1e-12);
    }
    // Table 4.22 counts houses insured, one a policy, whatever the term.
    let houses = werner_modlin(6, 1.0);
    let on = |y, m, d| prospicio_pricing::earned::in_force(&houses, Date::new(y, m, d).unwrap());
    assert_eq!([on(2011, 1, 1), on(2011, 6, 15), on(2012, 1, 1)], [2.0, 2.0, 2.0]);
}

fn date(y: i32, m: u8, d: u8) -> Date {
    Date::new(y, m, d).unwrap()
}

fn table(
    policies: &[Policy],
    windows: &Windows,
    kind: PeriodKind,
    method: EarningMethod,
) -> Vec<prospicio_pricing::earned::EarnedRow> {
    earned_exposure(policies, windows, kind, method, None)
        .unwrap()
        .rows()
        .to_vec()
}

#[test]
fn the_half_month_grid_reproduces_chapter_4_exactly() {
    // Tables 4.11 and 4.19: on the book's half-month grid a quarter of a
    // policy is a quarter, so the calendar year earned exposures are exact.
    let years = Windows::years(2010, 2012).unwrap();
    for (months, amount, earned) in [(12, 1.0, [0.25, 3.25, 2.50]), (6, 0.5, [0.25, 2.00, 0.75])] {
        let rows = table(
            &werner_modlin(months, amount),
            &years,
            PeriodKind::Calendar,
            EarningMethod::HalfMonth,
        );
        for (row, want) in rows.iter().zip(earned) {
            assert!((row.earned - want).abs() < 1e-12, "{} against {want}", row.earned);
        }
    }
}

#[test]
fn a_cancellation_is_written_back_when_it_happens() {
    // Chapter 4 and chapter 5: Policy D (07/01/11 to 06/30/12, premium
    // 400) cancelled on March 31, 2012, after 75% of its term, contributes
    // 1 exposure and 400 to calendar year 2011 and -0.25 and -100 to 2012.
    // By policy year it is all in 2011.
    let years = Windows::years(2011, 2012).unwrap();
    for amount in [1.0, 400.0] {
        let d = [Policy::new(date(2011, 7, 1), date(2012, 7, 1), amount)
            .unwrap()
            .cancelled(date(2012, 4, 1))
            .unwrap()];
        let calendar = table(&d, &years, PeriodKind::Calendar, EarningMethod::HalfMonth);
        assert_eq!(calendar[0].written, amount);
        assert!((calendar[1].written + 0.25 * amount).abs() < 1e-12);
        assert!((calendar[0].earned - 0.5 * amount).abs() < 1e-12);
        assert!((calendar[1].earned - 0.25 * amount).abs() < 1e-12);
        assert_eq!((calendar[1].unearned.abs() < 1e-12, calendar[1].in_force), (true, 0.0));
        let policy = table(&d, &years, PeriodKind::Policy, EarningMethod::HalfMonth);
        assert!((policy[0].written - 0.75 * amount).abs() < 1e-12);
        assert!((policy[0].earned - 0.75 * amount).abs() < 1e-12);
        assert_eq!(policy[1].written, 0.0);
    }
    // Chapter 3, Table 3.1, Policy B: 600 from 04/01/10, cancelled on
    // December 31, 2010: a second record of -0.25 exposure and -150,
    // dated 12/31/10, so calendar year 2010 has the net 450.
    let b = [Policy::new(date(2010, 4, 1), date(2011, 4, 1), 600.0)
        .unwrap()
        .cancelled(date(2011, 1, 1))
        .unwrap()];
    let years = Windows::years(2010, 2011).unwrap();
    let rows = table(&b, &years, PeriodKind::Calendar, EarningMethod::HalfMonth);
    assert!((rows[0].written - 450.0).abs() < 1e-12 && rows[1].written == 0.0);
    assert!((rows[0].earned - 450.0).abs() < 1e-12 && rows[1].earned == 0.0);
}

#[test]
fn a_mid_term_change_is_written_for_the_term_still_to_run() {
    // Chapter 3, Table 3.1, Policy C: 1,000 from 07/01/10; on January 1,
    // 2011 the full-term premium becomes 1,200. The records are -500 and
    // +600 on 01/01/11: 1,000 written in 2010 and 100 in 2011, and 500
    // then 600 earned.
    let c = [Policy::new(date(2010, 7, 1), date(2011, 7, 1), 1_000.0)
        .unwrap()
        .endorsed(date(2011, 1, 1), 1_200.0)
        .unwrap()];
    let years = Windows::years(2010, 2011).unwrap();
    let rows = table(&c, &years, PeriodKind::Calendar, EarningMethod::HalfMonth);
    assert_eq!(rows[0].written, 1_000.0);
    assert!((rows[1].written - 100.0).abs() < 1e-12);
    assert!((rows[0].earned - 500.0).abs() < 1e-12 && (rows[1].earned - 600.0).abs() < 1e-12);
    // In force at the close of 2010 the premium is still 1,000.
    assert_eq!((rows[0].in_force, rows[1].in_force), (1_000.0, 0.0));
    assert!((rows[0].unearned - 500.0).abs() < 1e-12);

    // Chapter 5: Policy D, 400 from 07/01/11, 800 full term from January
    // 1, 2012. The policyholder pays 600, and the in-force premium is 400
    // through December 31, 2011, and 800 from January 1, 2012.
    let d = [Policy::new(date(2011, 7, 1), date(2012, 7, 1), 400.0)
        .unwrap()
        .endorsed(date(2012, 1, 1), 800.0)
        .unwrap()];
    let years = Windows::years(2011, 2012).unwrap();
    let rows = table(&d, &years, PeriodKind::Policy, EarningMethod::HalfMonth);
    assert!((rows[0].written - 600.0).abs() < 1e-12 && (rows[0].earned - 600.0).abs() < 1e-12);
    let on = |y, m, day| prospicio_pricing::earned::in_force(&d, date(y, m, day));
    assert_eq!([on(2011, 7, 1), on(2011, 12, 31), on(2012, 1, 1), on(2012, 6, 30)], [400.0, 400.0, 800.0, 800.0]);
    assert_eq!(on(2012, 7, 1), 0.0);
}

#[test]
fn twenty_fourths_reproduce_tables_4_23_and_4_24() {
    // 240 exposures of annual policies written each month of 2010, taken
    // as written on the 15th whatever the day.
    let policies: Vec<Policy> = (1..=12u8)
        .map(|month| {
            let day = [1, 9, 28][usize::from(month) % 3];
            Policy::new(date(2010, month, day), date(2011, month, day), 240.0).unwrap()
        })
        .collect();
    // Table 4.24: earned in 2010 and in 2011, month by month.
    let years = Windows::years(2010, 2011).unwrap();
    for (k, policy) in policies.iter().enumerate() {
        let rows = table(
            std::slice::from_ref(policy),
            &years,
            PeriodKind::Calendar,
            EarningMethod::TwentyFourths,
        );
        let in_2010 = 230.0 - 20.0 * k as f64;
        assert!((rows[0].earned - in_2010).abs() < 1e-10, "month {}", k + 1);
        assert!((rows[1].earned - (240.0 - in_2010)).abs() < 1e-10, "month {}", k + 1);
    }
    let rows = table(&policies, &years, PeriodKind::Calendar, EarningMethod::TwentyFourths);
    assert!((rows[0].earned - 1_440.0).abs() < 1e-9 && (rows[1].earned - 1_440.0).abs() < 1e-9);
    assert_eq!((rows[0].written, rows[1].written), (2_880.0, 0.0));
    // Table 4.23: in force as of 07/01/10, 01/01/11 and 07/01/11.
    let halves =
        Windows::new(vec![date(2010, 1, 1), date(2010, 7, 1), date(2011, 1, 1), date(2011, 7, 1)])
            .unwrap();
    let rows = table(&policies, &halves, PeriodKind::Calendar, EarningMethod::TwentyFourths);
    let in_force: Vec<f64> = rows.iter().map(|r| r.in_force).collect();
    assert_eq!(in_force, [1_440.0, 2_880.0, 1_440.0]);
}

#[test]
fn changes_and_cancellations_match_a_day_by_day_count() {
    // A policy changed twice and cancelled, by exact days: each day earns
    // the full-term amount then in effect over the term's 366 days.
    let policy = Policy::new(date(2023, 11, 10), date(2024, 11, 10), 1_000.0)
        .unwrap()
        .endorsed(date(2024, 2, 1), 1_600.0)
        .unwrap()
        .endorsed(date(2024, 5, 20), 400.0)
        .unwrap()
        .cancelled(date(2024, 8, 3))
        .unwrap();
    let term = 366.0;
    let amount_on = |d: Date| {
        if d < date(2023, 11, 10) || d >= date(2024, 8, 3) {
            0.0
        } else if d < date(2024, 2, 1) {
            1_000.0
        } else if d < date(2024, 5, 20) {
            1_600.0
        } else {
            400.0
        }
    };
    let quarters = Windows::quarters(2023, 2024).unwrap();
    let rows = table(
        std::slice::from_ref(&policy),
        &quarters,
        PeriodKind::Calendar,
        EarningMethod::Daily,
    );
    let mut carried = 0.0;
    for row in &rows {
        let days = row.to.days_since(row.from);
        let earned: f64 = (0..days).map(|i| amount_on(row.from.plus_days(i)) / term).sum();
        assert!((row.earned - earned).abs() < 1e-9, "{}", row.from);
        // In force on the window's last day, at the amount then in effect.
        assert_eq!(row.in_force, amount_on(row.to.plus_days(-1)), "{}", row.from);
        carried += row.written - row.earned;
        assert!((row.unearned - carried).abs() < 1e-9, "{}", row.from);
    }
    // The written amounts: 1,000 at inception; +600 for the 283 days to
    // run on 1 February; -1,200 for the 174 to run on 20 May; and the 99
    // days not earned of 400 back on the last day in force, 2 August.
    let written: Vec<f64> = rows.iter().map(|r| r.written).collect();
    let want = [
        0.0,
        0.0,
        0.0,
        1_000.0,
        600.0 * 283.0 / term,
        -1_200.0 * 174.0 / term,
        -400.0 * 99.0 / term,
        0.0,
    ];
    for (got, want) in written.iter().zip(want) {
        assert!((got - want).abs() < 1e-9, "{got} against {want}");
    }
    // Net of everything, what is written is what is earned.
    let (w, e): (f64, f64) = rows.iter().fold((0.0, 0.0), |a, r| (a.0 + r.written, a.1 + r.earned));
    assert!((w - e).abs() < 1e-9 && rows.last().unwrap().unearned.abs() < 1e-9);
    assert_eq!(policy.cancellation(), Some(date(2024, 8, 3)));
    assert_eq!(policy.changes().len(), 2);
}

#[test]
fn bad_changes_and_cancellations_are_refused() {
    let policy = || Policy::new(date(2024, 1, 1), date(2025, 1, 1), 100.0).unwrap();
    assert!(policy().endorsed(date(2024, 1, 1), 50.0).is_err()); // not after the effective date
    assert!(policy().endorsed(date(2025, 1, 1), 50.0).is_err()); // not before the expiration
    assert!(policy().endorsed(date(2024, 6, 1), -1.0).is_err());
    let changed = policy().endorsed(date(2024, 6, 1), 50.0).unwrap();
    assert!(changed.clone().endorsed(date(2024, 6, 1), 60.0).is_err()); // not after the last change
    assert!(changed.clone().cancelled(date(2024, 6, 1)).is_err()); // not after the change
    let cancelled = changed.cancelled(date(2024, 9, 1)).unwrap();
    assert!(cancelled.clone().cancelled(date(2024, 10, 1)).is_err()); // already cancelled
    assert!(cancelled.clone().endorsed(date(2024, 9, 1), 10.0).is_err()); // not before the cancellation
    assert!(policy().cancelled(date(2025, 1, 1)).is_err() && policy().cancelled(date(2024, 1, 1)).is_err());
    assert!(!cancelled.in_force_at(date(2024, 9, 1)) && cancelled.in_force_at(date(2024, 8, 31)));
}
