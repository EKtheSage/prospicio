//! A pattern on another origin basis, and ages off or past its end
//! (decision 18 of `docs/design/reserving-v02.md`, issue #544). No
//! reference implementation averages a pattern over origin periods or reads
//! a pattern off the triangle's ages (chainladder-python 0.10.1's
//! `DevelopmentConstant` keeps the triangle's grain), so every check is by
//! hand or on a synthetic book.
//!
//! * A. `SelectedPattern::averaged_over_origins`: linear and exponential
//!   accident-month patterns averaged to accident quarters and years against
//!   their closed forms, the rule below the first age, and a synthetic book
//!   of accident months summed to accident years and quarters, whose chain
//!   ladder with the averaged pattern gives each origin the sum of its
//!   months' ultimates, on the plain triangle path and through its latest
//!   diagonal, an accident year in progress included.
//! * B. The plain triangle path: monthly patterns on annual triangles (RAA
//!   and GenIns) select the annual factors, to the bit for a cumulative
//!   pattern; a misaligned pattern and the data-dependent reading of sparse
//!   age-to-age factors, pinned; a pattern past the oldest age folded into a
//!   constant tail, with no tail, a constant one and an estimated one, the
//!   last equal to the diagonal route; and the errors.
//! * C. Origins past the pattern's end on a latest diagonal: the spread of
//!   the tail at one step per origin period (chainladder-python's 1.02538
//!   one year past the end), on half-year and quarterly grids, with no
//!   development past the end, an estimated tail carried as a constant, the
//!   expected-loss methods, a fitted development applied to an older book,
//!   and the errors.
//! * D. Mack, both bootstraps and the one-year views with a monthly pattern
//!   and with a pattern past the oldest age.

use prospicio_reserving::{
    Benktander, BornhuetterFerguson, CapeCod, ChainLadder, ChainLadderFit, Development,
    DevelopmentColumn, Error, Grain, Lag, LatestDiagonal, Long, Mack, MackBootstrap, Month,
    OdpBootstrap, OneYearMethod, PastPatternEnd, PatternKind, PatternWeight, SelectedPattern, Tail,
    TailBondy, TailConstant, TailCurve, Triangle,
};
use prospicio_validation::triangle;

fn close_to(found: f64, expected: f64, rel: f64) -> bool {
    found == expected || (found - expected).abs() <= rel * expected.abs()
}

#[track_caller]
fn assert_rel(found: f64, expected: f64, rel: f64, what: &str) {
    assert!(
        close_to(found, expected, rel),
        "{what}: {found} against {expected}"
    );
}

#[track_caller]
fn assert_all(found: &[f64], expected: &[f64], rel: f64, what: &str) {
    assert_eq!(found.len(), expected.len(), "{what}: lengths");
    for (k, (f, e)) in found.iter().zip(expected).enumerate() {
        assert_rel(*f, *e, rel, &format!("{what}, element {k}"));
    }
}

/// The two fits are equal to the bit, NaN where both are NaN.
#[track_caller]
fn same_bits(a: &ChainLadderFit, b: &ChainLadderFit) {
    assert_eq!(format!("{a:?}"), format!("{b:?}"));
}

/// The chain ladder with `pattern` fully selected and `tail`.
fn chain_ladder(pattern: SelectedPattern, tail: Tail) -> ChainLadder {
    ChainLadder {
        development: Development {
            pattern: Some(pattern),
            ..Default::default()
        },
        tail,
    }
}

fn percent(values: Vec<(Lag, f64)>) -> SelectedPattern {
    SelectedPattern::new(PatternKind::PercentDeveloped, values)
}

/// The averaged pattern's value at `age`.
fn at(p: &SelectedPattern, age: Lag) -> f64 {
    p.values
        .iter()
        .find(|v| v.0 == age)
        .unwrap_or_else(|| panic!("no value at {age}"))
        .1
}

/// The latest diagonal of a single-segment triangle, as the type's.
fn diagonal_of(tri: &Triangle) -> LatestDiagonal {
    LatestDiagonal::from_triangle(tri, "values", None).unwrap()
}

/// The single-segment `tri` restricted to the origins from `from`.
fn origins_from(tri: &Triangle, from: i32) -> Triangle {
    let ages = tri.development();
    let (mut origin, mut age, mut values) = (Vec::new(), Vec::new(), Vec::new());
    for (o, p) in tri.origins().iter().enumerate() {
        if p.start().year() < from {
            continue;
        }
        for (d, &a) in ages.iter().enumerate() {
            if let Some(v) = tri.get(0, 0, o, d) {
                origin.push(p.start());
                age.push(a);
                values.push(v);
            }
        }
    }
    Triangle::from_long(&Long {
        keys: &[],
        origin: &origin,
        development: DevelopmentColumn::Age(&age),
        values: &[("values", &values)],
        origin_grain: tri.origin_grain(),
        development_grain: tri.development_grain(),
        cumulative: true,
    })
    .unwrap()
}

/// chainladder-python's seed `x` of a tail `factor` spread at decay 0.5:
/// the positive root of `a x^2 + b x - ln(factor) = 0`, `a` and `b` the
/// sums of `0.25^k` and `0.5^k` over 1000 periods.
fn seed(factor: f64) -> f64 {
    let (mut a, mut b, mut d) = (0.0, 0.0, 1.0);
    for _ in 0..1000 {
        a += d * d;
        b += d;
        d *= 0.5;
    }
    (-b + (b * b + 4.0 * a * factor.ln()).sqrt()) / (2.0 * a)
}

/// What is left of a tail `factor` spread at decay 0.5 after `k` steps:
/// `factor / prod_{i<k} (1 + x 0.5^i)`.
fn left(factor: f64, k: usize) -> f64 {
    let x = seed(factor);
    (0..k).fold(factor, |r, i| r / (1.0 + x * 0.5f64.powi(i as i32)))
}

// A. Averaging over origin periods.

#[test]
fn a_linear_monthly_pattern_averages_to_its_closed_form() {
    let month = percent((1..=24).map(|t| (t, f64::from(t) / 24.0)).collect());
    // Accident quarters: (p(a) + p(a - 1) + p(a - 2)) / 3.
    let q = month
        .averaged_over_origins(Grain::Month, Grain::Quarter)
        .unwrap();
    assert_eq!(q.kind, PatternKind::PercentDeveloped);
    assert_eq!(q.values.first().unwrap().0, 1);
    assert_eq!(q.values.last().unwrap().0, 26);
    assert_rel(at(&q, 3), 1.0 / 12.0, 1e-15, "quarter at 3");
    for a in 3..=24 {
        assert_rel(at(&q, a), f64::from(a - 1) / 24.0, 1e-15, "quarter");
    }
    assert_rel(at(&q, 25), 71.0 / 72.0, 1e-15, "quarter at 25");
    assert_eq!(at(&q, 26), 1.0);
    // Accident years.
    let y = month
        .averaged_over_origins(Grain::Month, Grain::Year)
        .unwrap();
    assert_eq!(y.values.last().unwrap().0, 35);
    assert_rel(at(&y, 12), 13.0 / 48.0, 1e-15, "year at 12");
    assert_rel(at(&y, 24), 37.0 / 48.0, 1e-15, "year at 24");
    assert_eq!(at(&y, 35), 1.0);
    assert_rel(at(&y, 6), 21.0 / 288.0, 1e-15, "year in progress");
    // p(t) = t / 12 to 12 months.
    let short = percent((1..=12).map(|t| (t, f64::from(t) / 12.0)).collect())
        .averaged_over_origins(Grain::Month, Grain::Year)
        .unwrap();
    assert_rel(at(&short, 12), 78.0 / 144.0, 1e-15, "t/12 at 12");
    assert_rel(at(&short, 6), 21.0 / 144.0, 1e-15, "t/12 at 6");
    assert_rel(
        at(&short, 18),
        (7.0 + 45.0 / 12.0) / 12.0,
        1e-15,
        "t/12 at 18",
    );
    // p(t) = t / 60: P(a) = (a - 5.5) / 60 once a year has passed.
    let long = percent((1..=60).map(|t| (t, f64::from(t) / 60.0)).collect())
        .averaged_over_origins(Grain::Month, Grain::Year)
        .unwrap();
    for a in 12..=60 {
        assert_rel(at(&long, a), (f64::from(a) - 5.5) / 60.0, 1e-15, "t/60");
    }
}

/// 1 - 0.9^t at 1 to 60 months, an accident month's pattern.
fn exponential() -> SelectedPattern {
    percent((1..=60).map(|t| (t, 1.0 - 0.9f64.powi(t as i32))).collect())
}

#[test]
fn an_exponential_monthly_pattern_averages_to_its_closed_form() {
    let y = exponential()
        .averaged_over_origins(Grain::Month, Grain::Year)
        .unwrap();
    let closed = |a: Lag| {
        1.0 - 0.9f64.powi(a as i32) * (0.9f64.powi(-12) - 1.0) / (12.0 * (1.0 / 0.9 - 1.0))
    };
    for a in 12..=60 {
        assert_rel(at(&y, a), closed(a), 1e-14, "exponential");
    }
    assert!((at(&y, 12) - 0.461_822_152_360_75).abs() < 1e-13);
    assert!((at(&y, 24) - 0.848_002_679_946_90).abs() < 1e-13);
    let direct: f64 = (1..=6).map(|t| 1.0 - 0.9f64.powi(t)).sum::<f64>() / 12.0;
    assert_rel(at(&y, 6), direct, 1e-15, "in progress");
    assert!((at(&y, 6) - 0.148_580_75).abs() < 1e-8);
}

#[test]
fn below_the_first_age_the_share_is_linear_from_zero() {
    // x/24 given from 6 months only: below 6 it is linear from 0, so the
    // same as given from 1.
    let from_one = percent((1..=24).map(|t| (t, f64::from(t) / 24.0)).collect())
        .averaged_over_origins(Grain::Month, Grain::Year)
        .unwrap();
    let from_six = percent((6..=24).map(|t| (t, f64::from(t) / 24.0)).collect())
        .averaged_over_origins(Grain::Month, Grain::Year)
        .unwrap();
    assert_eq!(from_six.values.first().unwrap().0, 6);
    for &(a, v) in &from_six.values {
        assert_rel(v, at(&from_one, a), 1e-15, "from six");
    }
    // 30% at 3 months: an accident year at 3 months has (0.3 + 0.2 + 0.1)
    // / 12.
    let from_three = percent(vec![(3, 0.3), (12, 0.8), (24, 1.0)])
        .averaged_over_origins(Grain::Month, Grain::Year)
        .unwrap();
    assert_rel(at(&from_three, 3), 0.05, 1e-15, "from three");
}

#[test]
fn the_conversion_keeps_its_kind_weight_and_errors() {
    let month = exponential();
    // The same grain returns the pattern itself.
    assert_eq!(
        month
            .averaged_over_origins(Grain::Month, Grain::Month)
            .unwrap(),
        month
    );
    // Age-to-age factors average as their factors to ultimate.
    let ata = SelectedPattern::new(
        PatternKind::AgeToAge,
        (1..=59)
            .map(|t| {
                let f = (1.0 - 0.9f64.powi(t + 1)) / (1.0 - 0.9f64.powi(t));
                (t as Lag, f)
            })
            .collect(),
    );
    let cdf = SelectedPattern::new(
        PatternKind::AgeToUltimate,
        (1..=60)
            .map(|t: i32| (t as Lag, (1.0 - 0.9f64.powi(60)) / (1.0 - 0.9f64.powi(t))))
            .collect(),
    );
    let (a, c) = (
        ata.averaged_over_origins(Grain::Month, Grain::Quarter)
            .unwrap(),
        cdf.averaged_over_origins(Grain::Month, Grain::Quarter)
            .unwrap(),
    );
    assert_eq!(a.values.len(), c.values.len());
    for (x, y) in a.values.iter().zip(&c.values) {
        assert_eq!(x.0, y.0);
        assert_rel(x.1, y.1, 1e-14, "age-to-age against to ultimate");
    }
    // A weight on every factor carries over.
    let weighted = month
        .clone()
        .with_weight(PatternWeight::All(0.4))
        .averaged_over_origins(Grain::Month, Grain::Year)
        .unwrap();
    assert_eq!(weighted.weight, PatternWeight::All(0.4));
    // The errors.
    let reason = |r: Result<SelectedPattern, Error>| match r {
        Err(Error::Pattern { age, reason }) => (age, reason),
        other => panic!("{other:?} is not a pattern error"),
    };
    assert_eq!(
        reason(month.averaged_over_origins(Grain::Year, Grain::Quarter)).1,
        "the origin grain must be a whole multiple of the pattern's origin grain"
    );
    assert!(
        reason(
            month
                .clone()
                .with_weight(PatternWeight::ByAge(vec![(12, 0.5)]))
                .averaged_over_origins(Grain::Month, Grain::Year)
        )
        .1
        .starts_with("weights by age do not carry over")
    );
    let single = SelectedPattern::new(PatternKind::AgeToAge, vec![(12, 1.1)]);
    let (age, why) = reason(single.averaged_over_origins(Grain::Month, Grain::Year));
    assert_eq!(age, Some(12));
    assert!(
        why.starts_with("a single age-to-age factor has no span"),
        "{why}"
    );
    let long = percent(vec![(1, 0.1), (2400, 1.0)]);
    assert_eq!(
        reason(long.averaged_over_origins(Grain::Month, Grain::Year)),
        (
            None,
            "the averaged pattern needs more than 2,400 monthly ages"
        )
    );
    // The pattern's own errors.
    assert_eq!(
        reason(percent(vec![]).averaged_over_origins(Grain::Month, Grain::Year)).1,
        "the pattern has no values"
    );
    assert_eq!(
        reason(percent(vec![(3, 0.5), (3, 0.6)]).averaged_over_origins(Grain::Month, Grain::Year)),
        (Some(3), "the age is given twice")
    );
    assert_eq!(
        reason(percent(vec![(3, 0.5), (6, -1.0)]).averaged_over_origins(Grain::Month, Grain::Year)),
        (Some(6), "a percent developed must be finite and positive")
    );
    // One cumulative value (decision 20) is linear from 0 up to its age and
    // held past it: at 3 months (1/2 + 1/3 + 1/6) / 12 of a year's months
    // are developed, and from 14 months every month holds the value.
    let one = percent(vec![(3, 0.5)])
        .averaged_over_origins(Grain::Month, Grain::Year)
        .unwrap();
    assert_eq!(one.kind, PatternKind::PercentDeveloped);
    assert_eq!(one.values.first().map(|v| v.0), Some(3));
    assert_rel(one.values[0].1, 1.0 / 12.0, 1e-15, "one value at 3 months");
    assert_eq!(one.values.last(), Some(&(14, 0.5)));
    // The result's last age, the end plus eleven months, does not fit in
    // a Lag.
    assert_eq!(
        reason(
            percent(vec![(Lag::MAX - 5, 0.5), (Lag::MAX, 1.0)])
                .averaged_over_origins(Grain::Month, Grain::Year)
        ),
        (Some(Lag::MAX), "the age is too large")
    );
}

/// A book of accident months from January 2015 with ultimate `u(j)` for the
/// `j`-th month, each reported along `1 - 0.9^t` to 60 months and held
/// there, summed to origin periods of `grain` and valued at the end of each
/// development period of that grain to December 2019.
fn synthetic(grain: Grain, u: impl Fn(u32) -> f64) -> Triangle {
    let p = |t: i64| {
        if t <= 0 {
            0.0
        } else {
            1.0 - 0.9f64.powi(t.min(60) as i32)
        }
    };
    let g = grain.months();
    let start = Month::january(2015);
    let valuation = Month::new(2019, 12).unwrap();
    let (mut origin, mut age, mut values) = (Vec::new(), Vec::new(), Vec::new());
    for o in 0..60 / g {
        let begin = start.add_months(i64::from(o * g));
        let mut a = g;
        while begin.add_months(i64::from(a) - 1) <= valuation {
            let v: f64 = (0..g)
                .map(|j| u(o * g + j) * p(i64::from(a) - i64::from(j)))
                .sum();
            origin.push(begin);
            age.push(a);
            values.push(v);
            a += g;
        }
    }
    Triangle::from_long(&Long {
        keys: &[],
        origin: &origin,
        development: DevelopmentColumn::Age(&age),
        values: &[("values", &values)],
        origin_grain: grain,
        development_grain: grain,
        cumulative: true,
    })
    .unwrap()
}

#[test]
fn a_book_of_accident_months_projects_to_its_ultimates() {
    let month = exponential();
    for grain in [Grain::Year, Grain::Quarter] {
        let g = grain.months();
        let book = synthetic(grain, |_| 100.0);
        let pattern = month.averaged_over_origins(Grain::Month, grain).unwrap();
        let cl = chain_ladder(pattern, Tail::default());
        let want = vec![100.0 * f64::from(g); book.origins().len()];
        let plain = cl.fit(&book, "values").unwrap();
        assert_all(&plain.ultimate, &want, 1e-12, "plain path");
        let diagonal = cl.fit_diagonal(&diagonal_of(&book)).unwrap();
        assert_all(&diagonal.ultimate, &want, 1e-12, "diagonal");
    }
    // Accident years valued at June 2020: the year in progress, at six
    // months, projects to its twelve months' ultimate.
    let p = |t: i64| 1.0 - 0.9f64.powi(t.clamp(0, 60) as i32);
    let origins: Vec<Month> = (2015..=2020).map(Month::january).collect();
    let june = Month::new(2020, 6).unwrap();
    let latest: Vec<f64> = origins
        .iter()
        .map(|&o| {
            (0..12)
                .map(|j: i64| {
                    let age = june.months_since(o.add_months(j)) + 1;
                    100.0 * p(age)
                })
                .sum()
        })
        .collect();
    let d = LatestDiagonal::at_valuation(Grain::Year, &origins, june, &latest).unwrap();
    assert_eq!(d.ages(), [66, 54, 42, 30, 18, 6]);
    let cl = chain_ladder(
        month
            .averaged_over_origins(Grain::Month, Grain::Year)
            .unwrap(),
        Tail::default(),
    );
    assert_all(
        &cl.fit_diagonal(&d).unwrap().ultimate,
        &[1200.0; 6],
        1e-12,
        "June",
    );
    // Uneven monthly ultimates: each origin projects to the sum of its
    // months' reported values over the averaged share, which is not the
    // sum of their ultimates.
    let u = |j: u32| 100.0 + 10.0 * f64::from(j % 12);
    let book = synthetic(Grain::Year, u);
    let pattern = month
        .averaged_over_origins(Grain::Month, Grain::Year)
        .unwrap();
    let plain = chain_ladder(pattern.clone(), Tail::default())
        .fit(&book, "values")
        .unwrap();
    for (o, &ultimate) in plain.ultimate.iter().enumerate() {
        let age = 60 - 12 * o as Lag;
        let months = |f: &dyn Fn(u32) -> f64| (0..12u32).map(f).sum::<f64>();
        let reported = months(&|j| u(12 * o as u32 + j) * p(i64::from(age) - i64::from(j)));
        // The accident year's share reported, by hand: its months' shares
        // over twelve, independent of the conversion.
        let share = months(&|j| p(i64::from(age) - i64::from(j))) / 12.0;
        assert_rel(ultimate, reported / share, 1e-12, "uneven");
        // The miss against the sum of the months' ultimates, by hand.
        let truth = months(&|j| u(12 * o as u32 + j));
        let miss = reported / share - truth;
        assert!(miss.abs() > 1e-6 * truth, "{o}: the miss is {miss}");
        assert!(
            ((ultimate - truth) - miss).abs() <= 1e-12 * truth,
            "{o}: the miss {} against {miss}",
            ultimate - truth
        );
    }
}

// B. The plain triangle path.

/// `fit`'s factors to ultimate at its ages, as a pattern.
fn own_cdfs(fit: &ChainLadderFit) -> Vec<(Lag, f64)> {
    fit.development
        .development
        .iter()
        .copied()
        .zip(fit.cdf.iter().copied())
        .collect()
}

/// `cdf` at annual ages, at every month between them, ln-linear in age.
fn monthly(cdf: &[(Lag, f64)]) -> Vec<(Lag, f64)> {
    let mut out = Vec::new();
    for w in cdf.windows(2) {
        let ((b, cb), (c, cc)) = (w[0], w[1]);
        out.push((b, cb));
        for m in b + 1..c {
            let share = f64::from(m - b) / f64::from(c - b);
            out.push((m, (cb.ln() + share * (cc.ln() - cb.ln())).exp()));
        }
    }
    out.push(cdf[cdf.len() - 1]);
    out
}

#[test]
fn a_monthly_cumulative_pattern_is_the_annual_one_on_an_annual_triangle() {
    for name in ["raa", "genins"] {
        let tri = triangle(name);
        let own = ChainLadder::default().fit(&tri, "values").unwrap();
        let annual: Vec<(Lag, f64)> = own_cdfs(&own)
            .into_iter()
            .map(|(a, c)| (a, c * 1.02))
            .collect();
        let by_month = SelectedPattern::new(PatternKind::AgeToUltimate, monthly(&annual));
        let by_year = SelectedPattern::new(PatternKind::AgeToUltimate, annual);
        for tail in [Tail::default(), Tail::from(1.05)] {
            let m = chain_ladder(by_month.clone(), tail)
                .fit(&tri, "values")
                .unwrap();
            let y = chain_ladder(by_year.clone(), tail)
                .fit(&tri, "values")
                .unwrap();
            assert_eq!(m, y, "{name}, {tail:?}");
        }
        // Mack, fully selected, so process error only.
        let mack = |p: &SelectedPattern| {
            Mack {
                development: Development {
                    pattern: Some(p.clone()),
                    ..Default::default()
                },
                ..Default::default()
            }
            .fit(&tri, "values")
            .unwrap()
        };
        assert_eq!(mack(&by_month), mack(&by_year), "{name}: Mack");
        // A blended pattern has decision 12's closed form, the same either
        // way.
        let half = |p: &SelectedPattern| p.clone().with_weight(PatternWeight::All(0.5));
        assert_eq!(
            mack(&half(&by_month)),
            mack(&half(&by_year)),
            "{name}: blended"
        );
        // Both bootstraps draw the same at the same seed.
        let odp = |p: &SelectedPattern| {
            OdpBootstrap {
                pattern: Some(p.clone()),
                n_sims: 200,
                seed: 7,
                ..Default::default()
            }
            .fit(&tri, "values")
            .unwrap()
            .reserves
        };
        assert_eq!(
            odp(&by_month).draw_matrix(),
            odp(&by_year).draw_matrix(),
            "{name}: ODP"
        );
        let mack_boot = |p: &SelectedPattern| {
            MackBootstrap {
                development: Development {
                    pattern: Some(p.clone()),
                    ..Default::default()
                },
                n_sims: 200,
                seed: 7,
                ..Default::default()
            }
            .fit(&tri, "values")
            .unwrap()
            .reserves
        };
        assert_eq!(
            mack_boot(&by_month).draw_matrix(),
            mack_boot(&by_year).draw_matrix(),
            "{name}: Mack's bootstrap"
        );
        // In the one-year views, a monthly bootstrap pattern is the annual
        // method pattern.
        let method = OneYearMethod::ChainLadder(chain_ladder(by_year.clone(), Tail::default()));
        let cdr = |p: &SelectedPattern| {
            OdpBootstrap {
                pattern: Some(p.clone()),
                n_sims: 100,
                seed: 3,
                ..Default::default()
            }
            .one_year(&tri, "values", &method)
            .unwrap()
            .cdr
        };
        assert_eq!(cdr(&by_month).draw_matrix(), cdr(&by_year).draw_matrix());
        let mack_cdr = |p: &SelectedPattern| {
            MackBootstrap {
                development: Development {
                    pattern: Some(p.clone()),
                    ..Default::default()
                },
                n_sims: 100,
                seed: 3,
                ..Default::default()
            }
            .one_year(&tri, "values", &method)
            .unwrap()
            .cdr
        };
        assert_eq!(
            mack_cdr(&by_month).draw_matrix(),
            mack_cdr(&by_year).draw_matrix()
        );
    }
}

#[test]
fn monthly_age_to_age_factors_multiply_to_the_annual_ones() {
    let raa = triangle("raa");
    let g = 1.01f64;
    let ata = SelectedPattern::new(PatternKind::AgeToAge, (1..=119).map(|t| (t, g)).collect());
    let fit = chain_ladder(ata, Tail::default())
        .fit(&raa, "values")
        .unwrap();
    for s in &fit.development.selected {
        assert_rel(s.unwrap(), g.powi(12), 1e-15, "g^12");
    }
    assert_eq!(fit.pattern_tail, None);
    assert_eq!(fit.pattern_rescale, None);
    let cdf = SelectedPattern::new(
        PatternKind::AgeToUltimate,
        (1..=120).map(|t| (t, g.powi(120 - t as i32))).collect(),
    );
    let twin = chain_ladder(cdf, Tail::default())
        .fit(&raa, "values")
        .unwrap();
    assert_all(&fit.ultimate, &twin.ultimate, 1e-14, "to ultimate");
    // GenIns's factors spread over the months of each year, f^(1/12).
    let genins = triangle("genins");
    let own = ChainLadder::default().fit(&genins, "values").unwrap();
    let spread: Vec<(Lag, f64)> = own
        .ldf()
        .iter()
        .enumerate()
        .flat_map(|(k, &f)| (0..12).map(move |m| (12 + 12 * k as Lag + m, f.powf(1.0 / 12.0))))
        .collect();
    let cl = chain_ladder(
        SelectedPattern::new(PatternKind::AgeToAge, spread),
        Tail::default(),
    );
    let plain = cl.fit(&genins, "values").unwrap();
    assert_all(&plain.ultimate, &own.ultimate, 1e-13, "GenIns spread");
    let diagonal = cl.fit_diagonal(&diagonal_of(&genins)).unwrap();
    assert_all(
        &plain.ultimate,
        &diagonal.ultimate,
        1e-14,
        "against the diagonal",
    );
}

/// A triangle of three accident years at 12, 24 and 36 months.
fn small() -> Triangle {
    let origin = [2020, 2020, 2020, 2021, 2021, 2022].map(Month::january);
    Triangle::from_long(&Long {
        keys: &[],
        origin: &origin,
        development: DevelopmentColumn::Age(&[12, 24, 36, 12, 24, 12]),
        values: &[("values", &[100.0, 150.0, 165.0, 110.0, 170.0, 120.0])],
        origin_grain: Grain::Year,
        development_grain: Grain::Year,
        cumulative: true,
    })
    .unwrap()
}

#[test]
fn a_misaligned_pattern_is_read_between_its_ages() {
    // Factors to ultimate at 6, 18 and 30 months on a 12, 24, 36 triangle.
    let p = SelectedPattern::new(
        PatternKind::AgeToUltimate,
        vec![(6, 4.0), (18, 2.0), (30, 1.25)],
    );
    let fit = chain_ladder(p, Tail::default())
        .fit(&small(), "values")
        .unwrap();
    assert_rel(
        fit.development.selected[0].unwrap(),
        1.788_854_381_999_831_7,
        1e-15,
        "12-24",
    );
    assert_eq!(fit.development.selected[1], None);
    assert_rel(fit.ldf()[1], 165.0 / 150.0, 1e-15, "24-36 estimated");
    let c24 = 1.581_138_830_084_189_8;
    assert_rel(
        fit.pattern_rescale.unwrap(),
        c24 / fit.cdf[1],
        1e-15,
        "rescale",
    );
    assert_eq!(fit.pattern_tail, None);
}

#[test]
fn an_off_grid_age_changes_how_age_to_age_factors_are_read() {
    let raa = triangle("raa");
    let selected = |values: Vec<(Lag, f64)>| {
        chain_ladder(
            SelectedPattern::new(PatternKind::AgeToAge, values),
            Tail::default(),
        )
        .fit(&raa, "values")
        .unwrap()
        .development
        .selected
    };
    // With 18 months, the factors are read on their own ages: C(12) =
    // 1.98, C(18) = 1.32, C(36) = 1.2 and C(54) = 1, ln-linear between.
    let off = selected(vec![(12, 1.5), (18, 1.1), (36, 1.2)]);
    assert_rel(off[0].unwrap(), 1.548_420_173_184_550_8, 1e-15, "12-24");
    assert_rel(off[1].unwrap(), 1.065_602_236_766_610_7, 1e-15, "24-36");
    assert_rel(off[2].unwrap(), 1.129_243_234_657_234, 1e-15, "36-48");
    assert!(off[3..].iter().all(Option::is_none));
    // On the triangle's ages, each factor runs to the next age (decision
    // 12).
    let on = selected(vec![(12, 1.5), (36, 1.2)]);
    assert_eq!(on[0], Some(1.5));
    assert_eq!(on[1], None);
    assert_eq!(on[2], Some(1.2));
    assert!(on[3..].iter().all(Option::is_none));
}

/// RAA's own factors to ultimate times 1.1 at 12 to 120 months, and 1 at
/// 180: a pattern that runs past RAA's oldest age.
fn past_raa() -> SelectedPattern {
    let own = ChainLadder::default()
        .fit(&triangle("raa"), "values")
        .unwrap();
    let mut values: Vec<(Lag, f64)> = own_cdfs(&own)
        .into_iter()
        .map(|(a, c)| (a, c * 1.1))
        .collect();
    values.push((180, 1.0));
    SelectedPattern::new(PatternKind::AgeToUltimate, values)
}

#[test]
fn a_pattern_past_the_oldest_age_folds_into_a_constant_tail() {
    let raa = triangle("raa");
    let own = ChainLadder::default().fit(&raa, "values").unwrap();
    let fit = |tail: Tail| chain_ladder(past_raa(), tail).fit(&raa, "values");
    let none = fit(Tail::default()).unwrap();
    assert_rel(none.pattern_tail.unwrap(), 1.1, 1e-15, "no tail");
    assert_rel(none.tail.factor, 1.1, 1e-15, "no tail's factor");
    assert_eq!(none.tail.std_err, 0.0);
    assert_eq!(none.pattern_rescale, Some(1.0));
    assert_all(
        &none.ultimate,
        &own.ultimate.iter().map(|u| u * 1.1).collect::<Vec<_>>(),
        1e-14,
        "no tail",
    );
    // With no tail it is the pattern cut at the oldest age (decision 12
    // item 5).
    let cut = SelectedPattern {
        values: past_raa().values[..10].to_vec(),
        ..past_raa()
    };
    let truncated = chain_ladder(cut.clone(), Tail::default())
        .fit(&raa, "values")
        .unwrap();
    assert_all(&none.ultimate, &truncated.ultimate, 1e-15, "cut");
    // A constant tail follows the pattern's end.
    let constant = fit(Tail::from(1.05)).unwrap();
    assert_rel(constant.pattern_tail.unwrap(), 1.155, 1e-15, "constant");
    assert_rel(
        constant.pattern_rescale.unwrap(),
        1.0 / 1.05,
        1e-15,
        "rescale",
    );
    // An estimated tail is fitted on the pattern's own ages, as the
    // diagonal route fits it.
    let curve = Tail::from(TailCurve::default());
    let estimated = fit(curve).unwrap();
    let diagonal = chain_ladder(past_raa(), curve)
        .fit_diagonal(&diagonal_of(&raa))
        .unwrap();
    assert_rel(
        estimated.pattern_tail.unwrap(),
        1.1 * diagonal.tail.factor,
        1e-14,
        "curve",
    );
    assert_all(&estimated.ultimate, &diagonal.ultimate, 1e-14, "curve");
    // A tail attached at an age cannot follow the pattern's end.
    let attached = Tail::Constant(TailConstant {
        factor: 1.05,
        attachment_age: Some(120),
        ..Default::default()
    });
    assert_eq!(
        fit(attached).unwrap_err(),
        Error::Pattern {
            age: Some(120),
            reason: "a pattern that runs past the oldest age takes the tail at its end: leave \
                     the tail's attachment age unset"
        }
    );
    // The fold keeps a constant's decay: the same as the pattern cut at
    // the oldest age with the folded constant, 1.155, at that decay.
    let decayed = |factor: f64, decay: f64| {
        Tail::Constant(TailConstant {
            factor,
            decay,
            attachment_age: None,
        })
    };
    let folded = fit(decayed(1.05, 0.3)).unwrap();
    let cut_at = |decay: f64| {
        chain_ladder(cut.clone(), decayed(1.155, decay))
            .fit(&raa, "values")
            .unwrap()
    };
    assert_all(&folded.tail.ldf, &cut_at(0.3).tail.ldf, 1e-14, "decay 0.3");
    assert_all(&folded.ultimate, &cut_at(0.3).ultimate, 1e-14, "decay 0.3");
    let n = raa.development().len() - 1;
    assert!(
        (folded.tail.ldf[n] - cut_at(0.5).tail.ldf[n]).abs() > 1e-4,
        "the decay is not the default's"
    );
    // The fold takes the pattern at full weight: any weight below 1 is an
    // error, and weight 1 given explicitly is the default.
    let weighted = |weight: PatternWeight| {
        chain_ladder(past_raa().with_weight(weight), Tail::from(1.05)).fit(&raa, "values")
    };
    let full_weight = Error::Pattern {
        age: Some(120),
        reason: "a pattern that runs past the oldest age folds into the tail at full weight: \
                 give it weight 1, or cut it at the oldest age to blend it",
    };
    for weight in [
        PatternWeight::All(0.0),
        PatternWeight::All(0.5),
        PatternWeight::ByAge(vec![(12, 0.5)]),
    ] {
        assert_eq!(weighted(weight).unwrap_err(), full_weight);
    }
    assert_eq!(
        chain_ladder(
            past_raa().with_weight(PatternWeight::All(0.0)),
            Tail::default()
        )
        .fit(&raa, "values")
        .unwrap_err(),
        full_weight
    );
    assert_eq!(weighted(PatternWeight::All(1.0)).unwrap(), constant);
    assert_eq!(
        weighted(PatternWeight::ByAge(vec![(12, 1.0)])).unwrap(),
        constant
    );
}

#[test]
fn an_estimated_tail_on_a_finer_pattern_ending_at_the_oldest_age_depends_on_the_route() {
    // RAA's own factors to ultimate times 1.1, spread monthly to 120
    // months: on the plain path the curve is fitted on the annual links
    // the pattern selects, on the diagonal on its monthly factors
    // (decision 18 item 9).
    let raa = triangle("raa");
    let own = ChainLadder::default().fit(&raa, "values").unwrap();
    let annual: Vec<(Lag, f64)> = own_cdfs(&own)
        .into_iter()
        .map(|(a, c)| (a, c * 1.1))
        .collect();
    let by_month = SelectedPattern::new(PatternKind::AgeToUltimate, monthly(&annual));
    let curve = Tail::from(TailCurve::default());
    let plain = chain_ladder(by_month.clone(), curve)
        .fit(&raa, "values")
        .unwrap();
    let diagonal = chain_ladder(by_month, curve)
        .fit_diagonal(&diagonal_of(&raa))
        .unwrap();
    // The plain path is the annual pattern's.
    let annual_fit = chain_ladder(
        SelectedPattern::new(PatternKind::AgeToUltimate, annual),
        curve,
    )
    .fit(&raa, "values")
    .unwrap();
    assert_eq!(plain.tail.factor, annual_fit.tail.factor);
    assert_eq!(plain.pattern_tail, None);
    assert_rel(
        plain.tail.factor,
        PLAIN_CURVE,
        1e-12,
        "the plain path's curve",
    );
    assert_rel(
        diagonal.tail.factor,
        DIAGONAL_CURVE,
        1e-12,
        "the diagonal's curve",
    );
    assert!((plain.tail.factor - diagonal.tail.factor).abs() > 1e-3);
}

/// The curve tails of the monthly pattern to 120 months on RAA, recorded
/// from this build (Windows, 2026-10-23).
const PLAIN_CURVE: f64 = 1.009_435_751_581_231;
const DIAGONAL_CURVE: f64 = 1.012_188_851_544_037_8;

#[test]
fn the_plain_paths_pattern_errors() {
    let raa = triangle("raa");
    let error = |kind, values: Vec<(Lag, f64)>, weight| match chain_ladder(
        SelectedPattern::new(kind, values).with_weight(weight),
        Tail::default(),
    )
    .fit(&raa, "values")
    {
        Err(Error::Pattern { age, reason }) => (age, reason),
        other => panic!("{other:?} is not a pattern error"),
    };
    let all = PatternWeight::All(1.0);
    assert_eq!(
        error(PatternKind::AgeToAge, vec![(18, 1.5)], all.clone()),
        (Some(18), "not a development age of the triangle")
    );
    // One cumulative value (decision 20) is the tail from its age: off the
    // triangle's ages it selects nothing, and on them only at the oldest
    // age, with no tail and at full weight.
    assert_eq!(
        error(PatternKind::AgeToUltimate, vec![(18, 1.5)], all.clone()),
        (Some(18), "the pattern selects no factor of the triangle")
    );
    let one_value = "a cumulative pattern with one value is the tail from its age: give it at \
                     the oldest age, or give two or more ages";
    assert_eq!(
        error(PatternKind::AgeToUltimate, vec![(12, 1.5)], all.clone()),
        (Some(12), one_value)
    );
    assert_eq!(
        error(PatternKind::PercentDeveloped, vec![(108, 0.9)], all.clone()),
        (Some(108), one_value)
    );
    assert_eq!(
        error(
            PatternKind::AgeToUltimate,
            vec![(120, 1.05)],
            PatternWeight::All(0.5)
        ),
        (
            Some(120),
            "a cumulative pattern with one value is the tail, taken at full weight: give it \
             weight 1"
        )
    );
    let one = SelectedPattern::new(PatternKind::AgeToUltimate, vec![(120, 1.05)]);
    match chain_ladder(one.clone(), Tail::from(1.05)).fit(&raa, "values") {
        Err(Error::Pattern { age, reason }) => assert_eq!(
            (age, reason),
            (
                Some(120),
                "a cumulative pattern with one value is the tail, and a tail is given: give no \
                 tail, or two or more ages"
            )
        ),
        other => panic!("{other:?} is not a pattern error"),
    }
    // At the oldest age with no tail it is that constant tail, with no
    // parameter error.
    let fit = chain_ladder(one, Tail::default())
        .fit(&raa, "values")
        .unwrap();
    let tailed = ChainLadder {
        tail: Tail::from(1.05),
        ..Default::default()
    }
    .fit(&raa, "values")
    .unwrap();
    assert_eq!(fit.pattern_tail, Some(1.05));
    assert_eq!(fit.pattern_rescale, Some(1.0));
    assert_eq!(fit.tail.std_err, 0.0);
    assert_all(&fit.ultimate, &tailed.ultimate, 1e-15, "the tail of 1.05");
    assert_eq!(
        error(
            PatternKind::AgeToUltimate,
            vec![(13, 1.5), (23, 1.2)],
            all.clone()
        ),
        (Some(13), "the pattern selects no factor of the triangle")
    );
    assert_eq!(
        error(
            PatternKind::AgeToUltimate,
            vec![(12, 2.0), (18, 1.5), (36, 1.0)],
            PatternWeight::ByAge(vec![(18, 0.5)])
        ),
        (
            Some(18),
            "a weight is given at an age with no selected factor"
        )
    );
}

// C. Origins past the pattern's end on a latest diagonal.

const X: f64 = 0.024_010_738_379_690_81;

#[test]
fn the_seed_of_the_spread_is_chainladder_pythons() {
    assert_rel(seed(1.05), X, 1e-15, "x");
    // chainladder-python's TailConstant(1.05) on RAA: 1.02538 from 132
    // months.
    assert!((left(1.05, 1) - 1.02538).abs() < 5e-6);
    // The counterfactuals of decision 18 item 7: a monthly step leaves
    // this much of the tail one year past the end, and an origin a quarter
    // of an annual step past the end reads ln-linearly between 1.05 and
    // the first step's 1.02538.
    assert_rel(
        left(1.05, 12),
        1.001_160_241_303_754_1,
        1e-14,
        "monthly step",
    );
    assert_rel(
        1.05 * (1.0 + X).powf(-0.25),
        1.043_790_095_039_400_3,
        1e-15,
        "a quarter of an annual step",
    );
}

/// Ten age-to-age factors of 1.1 from 12 to 108 months, ending at 120.
fn to_120() -> SelectedPattern {
    SelectedPattern::new(
        PatternKind::AgeToAge,
        (0..9).map(|k| (12 + 12 * k, 1.1)).collect(),
    )
}

/// A diagonal of accident years from 2000 at `ages`, each latest value 100.
fn years(ages: &[Lag]) -> LatestDiagonal {
    let origins: Vec<Month> = (0..ages.len() as i32)
        .map(|k| Month::january(2000 + k))
        .collect();
    LatestDiagonal::new(Grain::Year, &origins, ages, &vec![100.0; ages.len()]).unwrap()
}

/// Each origin's factor to ultimate in `fit`, oldest first.
fn cdfs(fit: &ChainLadderFit) -> Vec<f64> {
    fit.ultimate
        .iter()
        .zip(&fit.latest)
        .map(|(u, l)| u / l)
        .collect()
}

#[test]
fn reject_names_the_settings() {
    let d = years(&[132, 120, 108]);
    let e = chain_ladder(to_120(), Tail::default())
        .fit_diagonal(&d)
        .unwrap_err();
    let Error::Diagonal(why) = e else {
        panic!("{e} is not a diagonal error")
    };
    assert!(why.starts_with(
        "origin 2000 is at 132 months, outside the pattern's ages from 12 to 120 months: an \
         origin older than the pattern's end needs LatestDiagonal::with_past_pattern_end with \
         PastPatternEnd::NoDevelopment (a factor of 1), or PastPatternEnd::Tail"
    ));
    assert!(
        why.ends_with("(TailCurve, TailBondy, Tail::LogLinear)"),
        "{why}"
    );
    // A younger origin keeps its message as it was.
    let young = LatestDiagonal::new(
        Grain::Year,
        &[2000, 2001].map(Month::january),
        &[24, 6],
        &[1.0, 1.0],
    )
    .unwrap()
    .with_past_pattern_end(PastPatternEnd::Tail);
    assert_eq!(
        chain_ladder(to_120(), Tail::from(1.05))
            .fit_diagonal(&young)
            .unwrap_err(),
        Error::Diagonal(
            "origin 2001 is at 6 months, outside the pattern's ages from 12 to 120 months".into()
        )
    );
    assert_eq!(d.past_pattern_end(), PastPatternEnd::Reject);
}

#[test]
fn the_tail_is_spread_at_one_step_per_origin_period() {
    let tail = Tail::from(1.05);
    let d = years(&[144, 132, 120, 108, 12]).with_past_pattern_end(PastPatternEnd::Tail);
    let fit = chain_ladder(to_120(), tail).fit_diagonal(&d).unwrap();
    let c = cdfs(&fit);
    assert_rel(c[0], 1.013_215_841_899_007_8, 1e-14, "144");
    assert_rel(c[1], 1.025_379_872_150_005_4, 1e-14, "132");
    assert_rel(c[0], left(1.05, 2), 1e-14, "144 by hand");
    assert_rel(c[1], left(1.05, 1), 1e-14, "132 by hand");
    assert_rel(c[2], 1.05, 1e-14, "120");
    assert_rel(c[3], 1.1 * 1.05, 1e-14, "108");
    assert_eq!(fit.pattern_tail, None);
    assert_eq!(fit.pattern_rescale, None);
    assert_rel(fit.tail.factor, left(1.05, 2), 1e-14, "the grid's tail");
    // Half a year past the end, on a half-year grid.
    let half = years(&[126, 114, 102, 90, 78, 66, 54, 42, 30, 18])
        .with_past_pattern_end(PastPatternEnd::Tail);
    let fit = chain_ladder(to_120(), tail).fit_diagonal(&half).unwrap();
    assert_rel(cdfs(&fit)[0], 1.037_616_916_668_914_9, 1e-14, "126");
    assert_rel(cdfs(&fit)[0], 1.05 / (1.0 + X).sqrt(), 1e-14, "126 by hand");
    // Accident quarters, a quarterly pattern to 120 months: one origin
    // period past the end is one step, whatever the pattern's grain.
    let quarterly = SelectedPattern::new(
        PatternKind::AgeToAge,
        (1..40).map(|k| (3 * k, 1.01)).collect(),
    );
    let origins: Vec<Month> = (0..41)
        .map(|k| Month::january(2000).add_months(3 * k))
        .collect();
    let ages: Vec<Lag> = (0..41).map(|k| 123 - 3 * k).collect();
    let q = LatestDiagonal::new(Grain::Quarter, &origins, &ages, &vec![100.0; 41])
        .unwrap()
        .with_past_pattern_end(PastPatternEnd::Tail);
    let fit = chain_ladder(quarterly, tail).fit_diagonal(&q).unwrap();
    assert_rel(cdfs(&fit)[0], 1.025_379_872_150_005_4, 1e-14, "123");
    assert_rel(cdfs(&fit)[1], 1.05, 1e-14, "120");
    // Accident years valued at March, the oldest at 123 months: a quarter
    // of an annual step past the end, read ln-linearly between 1.05 at 120
    // months and the first step's 1.02538 at 132. The decay is per origin
    // period, so the same age takes more of the tail than the accident
    // quarter above (decision 18 items 7 and 11).
    let march = years(&[123, 111, 99, 87, 75, 63, 51, 39, 27, 15])
        .with_past_pattern_end(PastPatternEnd::Tail);
    let fit = chain_ladder(to_120(), tail).fit_diagonal(&march).unwrap();
    assert_rel(cdfs(&fit)[0], 1.043_790_095_039_400_3, 1e-14, "March 123");
    assert_rel(
        cdfs(&fit)[0],
        1.05 * (1.0 + X).powf(-0.25),
        1e-14,
        "March 123 by hand",
    );
    // A monthly pattern equal to the annual one gives the same ultimates
    // past the end, since the spread follows the origins.
    let annual: Vec<(Lag, f64)> = (0..10)
        .map(|k| (12 + 12 * k, 1.1f64.powi(9 - k as i32)))
        .collect();
    let by_year = SelectedPattern::new(PatternKind::AgeToUltimate, annual.clone());
    let by_month = SelectedPattern::new(PatternKind::AgeToUltimate, monthly(&annual));
    let y = chain_ladder(by_year, tail).fit_diagonal(&d).unwrap();
    let m = chain_ladder(by_month, tail).fit_diagonal(&d).unwrap();
    assert_all(&m.ultimate, &y.ultimate, 1e-14, "monthly against annual");
    assert_rel(cdfs(&y)[1], left(1.05, 1), 1e-14, "annual 132");
}

#[test]
fn no_development_past_the_end_is_a_factor_of_one() {
    let d = years(&[132, 120, 108, 12]).with_past_pattern_end(PastPatternEnd::NoDevelopment);
    let fit = chain_ladder(to_120(), Tail::default())
        .fit_diagonal(&d)
        .unwrap();
    assert_eq!(fit.ultimate[0], fit.latest[0]);
    assert_rel(cdfs(&fit)[2], 1.1, 1e-15, "108");
    // A cumulative level of 95% at the end is rescaled to 100% for every
    // origin, past the end or not.
    let level = SelectedPattern::new(
        PatternKind::PercentDeveloped,
        vec![(12, 0.4), (60, 0.8), (120, 0.95)],
    );
    for d in [d.clone(), years(&[120, 108, 12])] {
        let d = d.with_past_pattern_end(PastPatternEnd::NoDevelopment);
        let fit = chain_ladder(level.clone(), Tail::default())
            .fit_diagonal(&d)
            .unwrap();
        assert_rel(
            fit.pattern_rescale.unwrap(),
            1.052_631_578_947_368_4,
            1e-15,
            "rescale",
        );
        assert_eq!(fit.pattern_tail, None);
        let n = fit.ultimate.len();
        assert_rel(fit.ultimate[n - 1], 100.0 / 0.4 * 0.95, 1e-14, "youngest");
    }
    // With no origin past the end, Reject keeps the level as the implied
    // tail (decision 16), so NoDevelopment moves the young origins too.
    let reject = chain_ladder(level, Tail::default())
        .fit_diagonal(&years(&[120, 108, 12]))
        .unwrap();
    assert_rel(reject.ultimate[2], 250.0, 1e-14, "youngest under Reject");
    assert_rel(
        reject.pattern_tail.unwrap(),
        1.0 / 0.95,
        1e-15,
        "the implied tail",
    );
    assert_eq!(reject.pattern_rescale, Some(1.0));
}

#[test]
fn an_estimated_tail_past_the_end_is_its_constant_spread() {
    // RAA's own factors, to 120 months, and two origins past the end.
    let raa = triangle("raa");
    let own = ChainLadder::default().fit(&raa, "values").unwrap();
    let pattern = SelectedPattern::new(
        PatternKind::AgeToAge,
        raa.development()
            .iter()
            .copied()
            .zip(own.ldf().iter().copied())
            .collect(),
    );
    let ages: Vec<Lag> = (0..12).map(|k| 144 - 12 * k).collect();
    for tail in [
        Tail::from(TailCurve::default()),
        Tail::from(TailBondy::default()),
        Tail::LogLinear,
    ] {
        // The factor fitted on the pattern's own ages, where no origin is
        // past the end.
        let fitted = chain_ladder(pattern.clone(), tail)
            .fit_diagonal(&diagonal_of(&raa))
            .unwrap()
            .tail
            .factor;
        assert!(fitted > 1.0, "{tail:?}");
        let d = years(&ages).with_past_pattern_end(PastPatternEnd::Tail);
        let estimated = chain_ladder(pattern.clone(), tail)
            .fit_diagonal(&d)
            .unwrap();
        let constant = chain_ladder(pattern.clone(), Tail::from(fitted))
            .fit_diagonal(&d)
            .unwrap();
        assert_all(&estimated.ultimate, &constant.ultimate, 1e-14, "estimated");
        assert_rel(cdfs(&estimated)[1], left(fitted, 1), 1e-14, "132");
    }
    // A single age-to-age factor spans one origin period, 12 to 24 months
    // here, and its estimated tail is fitted on that one link: Bondy's is
    // the factor itself, spread past 24 months; a curve needs two factors.
    let single = SelectedPattern::new(PatternKind::AgeToAge, vec![(12, 1.5)]);
    let d = years(&[36, 24, 12]).with_past_pattern_end(PastPatternEnd::Tail);
    let bondy = chain_ladder(single.clone(), Tail::from(TailBondy::default()))
        .fit_diagonal(&d)
        .unwrap();
    assert_all(
        &cdfs(&bondy),
        &[left(1.5, 1), 1.5, 2.25],
        1e-14,
        "a single factor's Bondy tail",
    );
    assert_eq!(
        chain_ladder(single, Tail::from(TailCurve::default()))
            .fit_diagonal(&d)
            .unwrap_err(),
        Error::Tail("a curve needs two factors above 1 in the fit period")
    );
}

/// `Tail`, and `NoDevelopment` on age-to-age factors (or a level of 100%),
/// change nothing when no origin is past the end; `NoDevelopment` on
/// another level does (`no_development_past_the_end_is_a_factor_of_one`).
#[test]
fn with_no_origin_past_the_end_the_settings_keep_an_age_to_age_fit() {
    let d = years(&[120, 108, 60, 12]);
    let reject = chain_ladder(to_120(), Tail::from(1.05))
        .fit_diagonal(&d)
        .unwrap();
    let tail = chain_ladder(to_120(), Tail::from(1.05))
        .fit_diagonal(&d.clone().with_past_pattern_end(PastPatternEnd::Tail))
        .unwrap();
    same_bits(&tail, &reject);
    let reject = chain_ladder(to_120(), Tail::default())
        .fit_diagonal(&d)
        .unwrap();
    let none = chain_ladder(to_120(), Tail::default())
        .fit_diagonal(&d.with_past_pattern_end(PastPatternEnd::NoDevelopment))
        .unwrap();
    same_bits(&none, &reject);
}

#[test]
fn the_settings_past_the_end_name_their_errors() {
    let d = years(&[132, 120, 12]);
    let diag = |e: Error| match e {
        Error::Diagonal(why) => why,
        other => panic!("{other} is not a diagonal error"),
    };
    let none = d
        .clone()
        .with_past_pattern_end(PastPatternEnd::NoDevelopment);
    assert_eq!(
        diag(
            chain_ladder(to_120(), Tail::from(1.05))
                .fit_diagonal(&none)
                .unwrap_err()
        ),
        "PastPatternEnd::NoDevelopment is a factor of 1 past the pattern's end: give no tail, \
         or use PastPatternEnd::Tail"
    );
    let tailed = d.with_past_pattern_end(PastPatternEnd::Tail);
    assert_eq!(
        diag(
            chain_ladder(to_120(), Tail::default())
                .fit_diagonal(&tailed)
                .unwrap_err()
        ),
        "PastPatternEnd::Tail needs a tail: for a factor of 1 use PastPatternEnd::NoDevelopment"
    );
    let attached = Tail::Constant(TailConstant {
        factor: 1.05,
        attachment_age: Some(120),
        ..Default::default()
    });
    assert_eq!(
        diag(
            chain_ladder(to_120(), attached)
                .fit_diagonal(&tailed)
                .unwrap_err()
        ),
        "a tail past the pattern's end attaches at the pattern's end: leave attachment_age unset"
    );
    // The spread's floor at decay 0.5, exp(-0.75) = 0.4723665527410147.
    assert!(matches!(
        chain_ladder(to_120(), Tail::from(0.4723)).fit_diagonal(&tailed),
        Err(Error::Tail(_))
    ));
    let above = chain_ladder(to_120(), Tail::from(0.4724))
        .fit_diagonal(&tailed)
        .unwrap();
    assert_rel(cdfs(&above)[1], 0.4724, 1e-14, "just above the floor");
}

#[test]
fn the_expected_loss_methods_take_the_spread() {
    let ages = [132, 108, 84, 60, 36, 12];
    let latest = [980.0, 860.0, 700.0, 520.0, 400.0, 150.0];
    let exposure = [1000.0; 6];
    let origins: Vec<Month> = (2019..=2024).map(Month::january).collect();
    let d = LatestDiagonal::new(Grain::Year, &origins, &ages, &latest)
        .unwrap()
        .with_exposure(&exposure)
        .unwrap()
        .with_past_pattern_end(PastPatternEnd::Tail);
    let chain_ladder = chain_ladder(to_120(), Tail::from(1.05));
    let cdf = cdfs(&chain_ladder.fit_diagonal(&d).unwrap());
    assert_rel(cdf[0], 1.025_379_872_150_005_4, 1e-14, "132");
    let elr = 0.7;
    let bf = BornhuetterFerguson {
        apriori: elr,
        chain_ladder: chain_ladder.clone(),
    }
    .fit_diagonal(&d)
    .unwrap();
    for o in 0..6 {
        let want = latest[o] + elr * exposure[o] * (1.0 - 1.0 / cdf[o]);
        assert_rel(bf.ultimate[o], want, 1e-14, "Bornhuetter-Ferguson");
    }
    assert_rel(
        bf.ultimate[0],
        980.0 + 700.0 * (1.0 - 1.0 / 1.025_379_872_150_005_4),
        1e-14,
        "Bornhuetter-Ferguson at 132",
    );
    let bk = Benktander {
        apriori: elr,
        n_iters: 2,
        chain_ladder: chain_ladder.clone(),
    }
    .fit_diagonal(&d)
    .unwrap();
    for o in 0..6 {
        let want = latest[o] + (1.0 - 1.0 / cdf[o]) * bf.ultimate[o];
        assert_rel(bk.ultimate[o], want, 1e-13, "Benktander");
    }
    let cc = CapeCod {
        chain_ladder,
        ..Default::default()
    }
    .fit_diagonal(&d)
    .unwrap();
    let used: f64 = exposure.iter().zip(&cdf).map(|(e, c)| e / c).sum();
    let ratio = latest.iter().sum::<f64>() / used;
    for o in 0..6 {
        let want = latest[o] + ratio * exposure[o] * (1.0 - 1.0 / cdf[o]);
        assert_rel(cc.expected_loss.ultimate[o], want, 1e-13, "Cape Cod");
    }
}

#[test]
fn a_younger_books_development_applies_to_older_origins() {
    // A source at most 60 months old applied to the full GenIns diagonal,
    // whose origins reach 120 months.
    let genins = triangle("genins");
    let young = origins_from(&genins, 2006);
    let d = diagonal_of(&genins);
    let none = ChainLadder::default().fit(&young, "values").unwrap();
    let fit = none
        .to_chain_ladder()
        .fit_diagonal(
            &d.clone()
                .with_past_pattern_end(PastPatternEnd::NoDevelopment),
        )
        .unwrap();
    for o in 0..5 {
        assert_rel(fit.ultimate[o], fit.latest[o], 1e-15, "no development");
    }
    assert_all(&fit.ultimate[5..], &none.ultimate, 1e-14, "the source's");
    let tailed = ChainLadder {
        tail: Tail::from(1.05),
        ..Default::default()
    }
    .fit(&young, "values")
    .unwrap();
    let fit = tailed
        .to_chain_ladder()
        .fit_diagonal(&d.with_past_pattern_end(PastPatternEnd::Tail))
        .unwrap();
    for o in 0..5 {
        // 2001 is at 120 months, five years past the source's 60.
        let k = 5 - o;
        assert_rel(
            fit.ultimate[o],
            fit.latest[o] * left(1.05, k),
            1e-14,
            "the tail",
        );
    }
    assert_all(&fit.ultimate[5..], &tailed.ultimate, 1e-14, "the source's");
}

// D. Mack, the bootstraps and the one-year views with a pattern past the
// oldest age.

/// `past_raa()` cut at RAA's oldest age, 120 months: its level there, 1.1,
/// is the implied tail (decision 12), the same constant the full pattern
/// folds into with no tail.
fn cut_raa() -> SelectedPattern {
    SelectedPattern {
        values: past_raa().values[..10].to_vec(),
        ..past_raa()
    }
}

#[test]
fn the_lifetime_views_keep_the_folded_tail_in_every_simulation() {
    let raa = triangle("raa");
    let odp = |pattern: SelectedPattern, tail: Tail, tail_std_err: Option<f64>| {
        OdpBootstrap {
            pattern: Some(pattern),
            tail,
            tail_std_err,
            n_sims: 200,
            seed: 11,
            ..Default::default()
        }
        .fit(&raa, "values")
    };
    let mack_boot = |pattern: SelectedPattern, tail: Tail| {
        MackBootstrap {
            development: Development {
                pattern: Some(pattern),
                ..Default::default()
            },
            tail,
            n_sims: 200,
            seed: 11,
            ..Default::default()
        }
        .fit(&raa, "values")
    };
    let draws = |r: Result<prospicio_prob::PredictiveDistribution, Error>| {
        r.unwrap().draw_matrix().to_vec()
    };
    // No tail: the folded 1.1 against the cut pattern's implied 1.1, the
    // same constant in every simulation, never refitted.
    let folded = draws(odp(past_raa(), Tail::default(), None).map(|f| f.reserves));
    let cut = draws(odp(cut_raa(), Tail::default(), None).map(|f| f.reserves));
    assert_all(&folded, &cut, 1e-13, "ODP");
    let folded = draws(mack_boot(past_raa(), Tail::default()).map(|f| f.reserves));
    let cut = draws(mack_boot(cut_raa(), Tail::default()).map(|f| f.reserves));
    assert_all(&folded, &cut, 1e-13, "Mack's bootstrap");
    // A constant tail folds to 1.155: the same as the cut pattern under a
    // constant tail of 1.155.
    let folded = draws(odp(past_raa(), Tail::from(1.05), None).map(|f| f.reserves));
    let cut = draws(odp(cut_raa(), Tail::from(1.155), None).map(|f| f.reserves));
    assert_all(&folded, &cut, 1e-13, "ODP, a constant tail");
    // Mack's bootstrap draws a given constant with Mack's extrapolated
    // standard error, and the fold, whose standard error is 0, fixed.
    let folded = draws(mack_boot(past_raa(), Tail::from(1.05)).map(|f| f.reserves));
    let cut = draws(mack_boot(cut_raa(), Tail::from(1.155)).map(|f| f.reserves));
    assert_ne!(folded, cut, "Mack's bootstrap draws the cut's tail error");
    // Mack: the fold carries no parameter error, as the implied tail.
    let mack = |pattern: SelectedPattern, tail: Tail| {
        Mack {
            development: Development {
                pattern: Some(pattern),
                ..Default::default()
            },
            tail,
            ..Default::default()
        }
        .fit(&raa, "values")
        .unwrap()
    };
    let (folded, cut) = (
        mack(past_raa(), Tail::default()),
        mack(cut_raa(), Tail::default()),
    );
    assert_eq!(folded.chain_ladder.tail.std_err, 0.0);
    assert_eq!(cut.chain_ladder.tail.std_err, 0.0);
    assert_all(&folded.standard_error, &cut.standard_error, 1e-13, "Mack");
    assert_rel(
        folded.total_standard_error,
        cut.total_standard_error,
        1e-13,
        "Mack's total",
    );
    // A tail standard error is read against the tail given: an estimated
    // tail's is still an error when the pattern folds it into a constant.
    let curve = Tail::from(TailCurve::default());
    assert_eq!(
        odp(past_raa(), curve, Some(0.01)).unwrap_err(),
        Error::Tail(
            "an estimated tail's parameter error comes from refitting it on each simulation: a \
             standard error is for a constant tail"
        )
    );
    // A constant's standard error is its own: the folded factor is drawn at
    // the constant's coefficient of variation, 0.02 / 1.05, which is the
    // cut pattern's under 1.155 with a standard error of 0.022.
    let folded = draws(odp(past_raa(), Tail::from(1.05), Some(0.02)).map(|f| f.reserves));
    let cut = draws(odp(cut_raa(), Tail::from(1.155), Some(0.022)).map(|f| f.reserves));
    assert_all(&folded, &cut, 1e-12, "ODP, the constant's standard error");
    let fixed = draws(odp(past_raa(), Tail::from(1.05), None).map(|f| f.reserves));
    assert_ne!(folded, fixed, "the standard error is drawn");
}

#[test]
fn the_bootstraps_take_a_one_value_pattern_as_its_tail() {
    // Decision 20: `{120: 1.05}` on RAA, the oldest age, is the implied
    // tail 1.05, a constant the ODP bootstrap draws as a given one, and
    // the one-year views take it from the method as any implied tail.
    let raa = triangle("raa");
    let one = || SelectedPattern::new(PatternKind::AgeToUltimate, vec![(120, 1.05)]);
    let odp = |pattern: Option<SelectedPattern>, tail: Tail| OdpBootstrap {
        pattern,
        tail,
        n_sims: 200,
        seed: 11,
        ..Default::default()
    };
    let by_value = odp(Some(one()), Tail::default())
        .fit(&raa, "values")
        .unwrap();
    let by_tail = odp(None, Tail::from(1.05)).fit(&raa, "values").unwrap();
    assert_eq!(by_value.chain_ladder.pattern_tail, Some(1.05));
    assert_all(
        by_value.reserves.draw_matrix(),
        by_tail.reserves.draw_matrix(),
        1e-13,
        "ODP, one value against its tail",
    );
    let method = OneYearMethod::ChainLadder(chain_ladder(one(), Tail::default()));
    let year = odp(None, Tail::default())
        .one_year(&raa, "values", &method)
        .unwrap();
    assert_eq!(year.bootstrap.chain_ladder.pattern_tail, Some(1.05));
    let year = MackBootstrap {
        n_sims: 50,
        seed: 4,
        ..Default::default()
    }
    .one_year(&raa, "values", &method)
    .unwrap();
    assert_eq!(year.bootstrap.mack.chain_ladder.pattern_tail, Some(1.05));
    // With a tail of its own the method's pattern and tail conflict
    // (decision 20 item 3), and the bootstrap, which reads the method's
    // pattern without its level, names that conflict (item 5).
    let tailed = OneYearMethod::ChainLadder(chain_ladder(one(), Tail::from(1.05)));
    match odp(None, Tail::default()).one_year(&raa, "values", &tailed) {
        Err(Error::Bootstrap(reason)) => {
            assert!(reason.contains("give the method no tail"), "{reason}")
        }
        other => panic!("{:?} is not a bootstrap error", other.err()),
    }
}

#[test]
fn the_one_year_views_carry_the_folded_tail() {
    let raa = triangle("raa");
    let method = |tail: Tail| OneYearMethod::ChainLadder(chain_ladder(past_raa(), tail));
    let odp = |tail: Tail| OdpBootstrap {
        n_sims: 50,
        seed: 4,
        tail,
        ..Default::default()
    };
    let mack = |tail: Tail| MackBootstrap {
        n_sims: 50,
        seed: 4,
        tail,
        ..Default::default()
    };
    // No tail on either side: the bootstrap takes the pattern and its
    // folded tail.
    let one = odp(Tail::default())
        .one_year(&raa, "values", &method(Tail::default()))
        .unwrap();
    assert_rel(
        one.bootstrap.chain_ladder.pattern_tail.unwrap(),
        1.1,
        1e-15,
        "ODP",
    );
    // Its draws are the cut pattern's, whose implied tail is the same 1.1.
    let cut = OneYearMethod::ChainLadder(chain_ladder(cut_raa(), Tail::default()));
    let twin = odp(Tail::default()).one_year(&raa, "values", &cut).unwrap();
    assert_all(
        one.cdr.draw_matrix(),
        twin.cdr.draw_matrix(),
        1e-12,
        "ODP's claims development result",
    );
    let one = mack(Tail::default())
        .one_year(&raa, "values", &method(Tail::default()))
        .unwrap();
    assert_rel(
        one.bootstrap.mack.chain_ladder.pattern_tail.unwrap(),
        1.1,
        1e-15,
        "Mack",
    );
    let twin = mack(Tail::default())
        .one_year(&raa, "values", &cut)
        .unwrap();
    assert_all(
        one.cdr.draw_matrix(),
        twin.cdr.draw_matrix(),
        1e-12,
        "Mack's claims development result",
    );
    // A method tail and none on the bootstrap: the level-free reading
    // cannot carry the method's tail into the pattern.
    let level_free = Error::Bootstrap(
        "a pattern that runs past the oldest age carries the re-reserving method's tail into \
         it: give the bootstrap the method's tail",
    );
    let tailed = method(Tail::from(1.05));
    assert_eq!(
        odp(Tail::default())
            .one_year(&raa, "values", &tailed)
            .unwrap_err(),
        level_free
    );
    assert_eq!(
        mack(Tail::default())
            .one_year(&raa, "values", &tailed)
            .unwrap_err(),
        level_free
    );
    // The same tail on both sides folds the same constant.
    let one = odp(Tail::from(1.05))
        .one_year(&raa, "values", &tailed)
        .unwrap();
    assert_rel(one.bootstrap.chain_ladder.tail.factor, 1.155, 1e-15, "ODP");
    let one = mack(Tail::from(1.05))
        .one_year(&raa, "values", &tailed)
        .unwrap();
    assert_rel(
        one.bootstrap.mack.chain_ladder.tail.factor,
        1.155,
        1e-15,
        "Mack",
    );
    // Merz and Wuthrich's closed form has no tail, and the fold is one.
    let fit = Mack {
        development: Development {
            pattern: Some(past_raa()),
            ..Default::default()
        },
        ..Default::default()
    }
    .fit(&raa, "values")
    .unwrap();
    assert_rel(fit.chain_ladder.tail.factor, 1.1, 1e-15, "Mack's tail");
    assert_eq!(
        fit.claims_development_result().unwrap_err(),
        Error::ClaimsDevelopment("needs no tail factor")
    );
}
