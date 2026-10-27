//! An accident-period development pattern on an underwriting basis
//! (decision 21 of `docs/design/reserving-v02.md`, issue #633). No
//! reference implementation converts a reserving pattern from accident to
//! underwriting periods, so the checks are by hand, against the pricing
//! crate's monthly cash-flow patterns, and on a simulated book.
//!
//! * A. The parallelogram by hand: a linear accident-month pattern under
//!   even writing over 12 months and 12-month policies, against its closed
//!   forms; the accident-year route equal to the accident-month route when
//!   the accident-year pattern is given at every month; the weight, the
//!   lengthening and the errors.
//! * B. A cross-check against `prospicio_pricing::cashflow::MonthlyPatterns::
//!   from_annual`, which convolves a written pattern with even earning and
//!   then with a payment pattern by accident age.
//! * C. A book of policies written every month, summed to an
//!   accident-year and an underwriting-year triangle: the conversion of the
//!   accident-year (or accident-month) chain ladder pattern is the
//!   underwriting-year chain ladder pattern, and projects each
//!   underwriting year to its ultimate; on the triangles valued annually,
//!   the misses of decision 18's reading (`AnnualReading::LnLinear`) and of
//!   the default Weibull reading (decision 25), pinned.
//! * D. The default reading against the design study's scipy prototype
//!   (`validation/scripts/ay_monthly_reading.py`, `wbl+exact` and
//!   `llog+exact`) on the book and the CAS loss reserve database's private
//!   passenger auto paid pattern.

use prospicio_pricing::cashflow::MonthlyPatterns;
use prospicio_pricing::payout::{Interpolation, PaymentPattern};
use prospicio_reserving::{
    AnnualReading, ChainLadder, ChainLadderFit, Development, DevelopmentColumn, Error, Grain,
    GrowthCurve, Lag, Long, Month, PatternKind, PatternWeight, SelectedPattern, Tail, Triangle,
    UnderwritingBasis,
};

fn assert_close(found: f64, expected: f64, tol: f64, what: &str) {
    assert!(
        (found - expected).abs() <= tol * expected.abs().max(1.0),
        "{what}: {found} against {expected}"
    );
}

/// The pattern's value at `age`.
fn at(p: &SelectedPattern, age: Lag) -> f64 {
    p.values
        .iter()
        .find(|v| v.0 == age)
        .unwrap_or_else(|| panic!("no value at {age}"))
        .1
}

fn percent(values: Vec<(Lag, f64)>) -> SelectedPattern {
    SelectedPattern::new(PatternKind::PercentDeveloped, values)
}

/// An accident month reporting `t / 60` of its ultimate at `t` months.
fn linear() -> SelectedPattern {
    percent((1..=60).map(|t| (t, f64::from(t) / 60.0)).collect())
}

/// The chain ladder with `pattern` fully selected and no tail.
fn chain_ladder(pattern: SelectedPattern) -> ChainLadder {
    ChainLadder {
        development: Development {
            pattern: Some(pattern),
            ..Default::default()
        },
        tail: Tail::default(),
    }
}

fn reason(e: Error) -> &'static str {
    match e {
        Error::Pattern { reason, .. } => reason,
        other => panic!("not a pattern error: {other}"),
    }
}

// A. The parallelogram by hand.

#[test]
fn the_parallelogram_of_a_linear_pattern_is_its_closed_form() {
    let e = UnderwritingBasis::default().exposure().unwrap();
    assert_eq!(e.len(), 24);
    for (t, &x) in e.iter().enumerate() {
        let t = t as f64;
        let want = if t <= 11.0 {
            2.0 * t + 1.0
        } else {
            47.0 - 2.0 * t
        } / 288.0;
        assert_close(x, want, 1e-15, "exposure");
    }
    let uy = linear()
        .to_underwriting_basis(Grain::Month, &UnderwritingBasis::default())
        .unwrap();
    assert_eq!(uy.kind, PatternKind::PercentDeveloped);
    assert_eq!(uy.values.first().map(|v| v.0), Some(1));
    // 60 months and 23 more: every accident month of the year past 60.
    assert_eq!(uy.values.last(), Some(&(83, 1.0)));
    for a in 1..=12u32 {
        // sum_{t < a} (2t + 1)(a - t) = a(a + 1)(2a + 1)/6, over 288 * 60.
        let a64 = f64::from(a);
        let want = a64 * (a64 + 1.0) * (2.0 * a64 + 1.0) / (1728.0 * 60.0);
        assert_close(at(&uy, a), want, 1e-14, "first year");
    }
    for a in 23..=60u32 {
        // The exposure's mean is 11.5 months.
        assert_close(at(&uy, a), (f64::from(a) - 11.5) / 60.0, 1e-14, "middle");
    }
    assert_close(at(&uy, 12), 3900.0 / 103_680.0, 1e-14, "12 months");
    assert_close(at(&uy, 36), 24.5 / 60.0, 1e-14, "36 months");
}

#[test]
fn an_accident_year_pattern_at_every_month_gives_the_accident_month_route() {
    let month = linear();
    let year = month
        .averaged_over_origins(Grain::Month, Grain::Year)
        .unwrap();
    let seasonal: Vec<f64> = (0..12).map(|j| 1.0 + f64::from(j % 4)).collect();
    for basis in [
        // The term a whole multiple of the year.
        UnderwritingBasis::default(),
        UnderwritingBasis::new(seasonal.clone(), 24),
        // The writing even within the year.
        UnderwritingBasis::even(12, 6),
        UnderwritingBasis::even(24, 7),
    ] {
        let direct = month.to_underwriting_basis(Grain::Month, &basis).unwrap();
        let by_year = year.to_underwriting_basis(Grain::Year, &basis).unwrap();
        assert_eq!(direct.values.len(), by_year.values.len(), "{basis:?}");
        assert_eq!(
            direct.values.last().map(|v| v.0),
            Some(60 + basis.writing.len() as Lag + basis.policy_months - 1)
        );
        for (d, y) in direct.values.iter().zip(&by_year.values) {
            assert_eq!(d.0, y.0);
            assert_close(y.1, d.1, 1e-14, "accident-year route");
        }
    }
    // Quarterly writing even within each quarter, and a quarterly pattern.
    let quarterly: Vec<f64> = (0..12).map(|j| [1.0, 3.0, 2.0, 4.0][j / 3]).collect();
    let quarter = month
        .averaged_over_origins(Grain::Month, Grain::Quarter)
        .unwrap();
    let basis = UnderwritingBasis::new(quarterly.clone(), 5);
    let direct = month.to_underwriting_basis(Grain::Month, &basis).unwrap();
    let by_quarter = quarter
        .to_underwriting_basis(Grain::Quarter, &basis)
        .unwrap();
    for (d, q) in direct.values.iter().zip(&by_quarter.values) {
        assert_close(q.1, d.1, 1e-14, "accident-quarter route");
    }
    // Seasonal writing with a term that is no whole number of years has no
    // exact accident-year route.
    let e = year
        .to_underwriting_basis(Grain::Year, &UnderwritingBasis::new(seasonal, 6))
        .unwrap_err();
    assert!(reason(e).contains("give an accident-month pattern"));
    // Uneven within a quarter on a quarterly pattern: the same error.
    let mut uneven = quarterly;
    uneven[1] *= 2.0;
    assert!(
        quarter
            .to_underwriting_basis(Grain::Quarter, &UnderwritingBasis::new(uneven, 5))
            .is_err()
    );
}

#[test]
fn the_conversion_keeps_its_kind_weight_and_errors() {
    let basis = UnderwritingBasis::default();
    // Age-to-age factors and factors to ultimate convert as their percents
    // developed: the annual factors of t/60 to 60 months.
    let cdf: Vec<(Lag, f64)> = (1..=5).map(|k| (12 * k, 5.0 / f64::from(k))).collect();
    let ata: Vec<(Lag, f64)> = (1..5)
        .map(|k| (12 * k, f64::from(k + 1) / f64::from(k)))
        .collect();
    let pd: Vec<(Lag, f64)> = (1..=5).map(|k| (12 * k, f64::from(k) / 5.0)).collect();
    // Under decision 18's reading to rounding; under the curve's, the
    // inputs' last bits move its least-squares search a little.
    for (reading, tol) in [
        (AnnualReading::LnLinear, 1e-14),
        (AnnualReading::default(), 1e-10),
    ] {
        let basis = basis.clone().with_reading(reading);
        let want = percent(pd.clone())
            .to_underwriting_basis(Grain::Year, &basis)
            .unwrap();
        for (kind, values) in [
            (PatternKind::AgeToUltimate, cdf.clone()),
            (PatternKind::AgeToAge, ata.clone()),
        ] {
            let got = SelectedPattern::new(kind, values)
                .to_underwriting_basis(Grain::Year, &basis)
                .unwrap();
            assert_eq!(got.values.len(), want.values.len());
            for (g, w) in got.values.iter().zip(&want.values) {
                assert_close(g.1, w.1, tol, "kind");
            }
        }
    }
    let want = percent(pd)
        .to_underwriting_basis(Grain::Year, &basis)
        .unwrap();
    // The curve's reading starts at 1 month.
    assert_eq!(want.values[0].0, 1);
    // The end: 60 months of an accident-year pattern and 12 more.
    assert_eq!(want.values.last(), Some(&(72, 1.0)));
    // A level below 1 is kept at the end.
    let level = percent(vec![(12, 0.4), (24, 0.9)])
        .to_underwriting_basis(Grain::Year, &basis)
        .unwrap();
    assert_eq!(level.values.last(), Some(&(36, 0.9)));
    // One weight carries over.
    let w = linear()
        .with_weight(PatternWeight::All(0.25))
        .to_underwriting_basis(Grain::Month, &basis)
        .unwrap();
    assert_eq!(w.weight, PatternWeight::All(0.25));
    // Weights by age do not.
    let e = linear()
        .with_weight(PatternWeight::ByAge(vec![(12, 0.5)]))
        .to_underwriting_basis(Grain::Month, &basis)
        .unwrap_err();
    assert!(reason(e).contains("give one weight for every factor"));
    let e = SelectedPattern::new(PatternKind::AgeToAge, vec![(12, 1.5)])
        .to_underwriting_basis(Grain::Year, &basis)
        .unwrap_err();
    assert!(reason(e).contains("no span to convert"));
    for (bad, text) in [
        (
            UnderwritingBasis::new(vec![1.0; 12], 0),
            "at least one month",
        ),
        (UnderwritingBasis::new(vec![], 12), "no months"),
        (UnderwritingBasis::new(vec![1.0, -1.0], 12), "not negative"),
        (
            UnderwritingBasis::new(vec![1.0, f64::NAN], 12),
            "not negative",
        ),
        (UnderwritingBasis::new(vec![0.0; 3], 12), "positive sum"),
    ] {
        let e = linear()
            .to_underwriting_basis(Grain::Month, &bad)
            .unwrap_err();
        assert!(reason(e).contains(text), "{bad:?}");
    }
    let e = percent(vec![(12, 0.5), (2400, 1.0)])
        .to_underwriting_basis(Grain::Year, &basis)
        .unwrap_err();
    assert!(reason(e).contains("2,400"));
    // Shares are scaled, and trailing zeros dropped.
    let mut padded = vec![2.0; 12];
    padded.extend([0.0; 6]);
    let scaled = linear()
        .to_underwriting_basis(Grain::Month, &UnderwritingBasis::new(padded, 12))
        .unwrap();
    let plain = linear()
        .to_underwriting_basis(Grain::Month, &basis)
        .unwrap();
    assert_eq!(scaled.values.len(), plain.values.len());
    for (s, p) in scaled.values.iter().zip(&plain.values) {
        assert_close(s.1, p.1, 1e-15, "scaled");
    }
}

#[test]
fn the_writing_fits_in_the_triangles_origin_period() {
    // A period's policies are written within it: the default's year of
    // writing fits underwriting years, and shorter periods do not hold it.
    let year = UnderwritingBasis::default();
    year.check_within(Grain::Year).unwrap();
    for grain in [Grain::Semester, Grain::Quarter, Grain::Month] {
        let e = year.check_within(grain).unwrap_err();
        assert!(reason(e).contains("longer than the triangle's origin period"));
    }
    UnderwritingBasis::even(3, 12)
        .check_within(Grain::Quarter)
        .unwrap();
    UnderwritingBasis::new(vec![1.0], 12)
        .check_within(Grain::Month)
        .unwrap();
    // Trailing zeros are dropped before the count; leading zeros count.
    UnderwritingBasis::new(vec![1.0, 1.0, 1.0, 0.0, 0.0], 12)
        .check_within(Grain::Quarter)
        .unwrap();
    assert!(
        UnderwritingBasis::new(vec![0.0, 0.0, 1.0, 1.0], 12)
            .check_within(Grain::Quarter)
            .is_err()
    );
    // The basis's own errors come first.
    let e = UnderwritingBasis::new(vec![], 12)
        .check_within(Grain::Year)
        .unwrap_err();
    assert!(reason(e).contains("no months"));
}

// B. Against the pricing crate's monthly patterns.

#[test]
fn the_conversion_is_the_cash_flow_engines_paid_pattern() {
    // Paid 40%, 30%, 20% and 10% over the four years from the accident
    // date, evenly within each.
    let paid = PaymentPattern::new(vec![1.0, 2.0, 3.0, 4.0], vec![0.4, 0.3, 0.2, 0.1]).unwrap();
    // An accident month at age k is on average k - 1/2 months from its
    // accident dates, as `from_annual` places losses mid-month.
    let month = percent(
        (1..=60)
            .map(|k| {
                (
                    k,
                    paid.paid_by((f64::from(k) - 0.5) / 12.0, Interpolation::Linear),
                )
            })
            .collect(),
    );
    let seasonal: Vec<f64> = (0..12).map(|j| 1.0 + 0.5 * f64::from(j % 5)).collect();
    for basis in [
        UnderwritingBasis::default(),
        UnderwritingBasis::new(seasonal.clone(), 12),
        UnderwritingBasis::new(seasonal, 7),
        UnderwritingBasis::even(3, 1),
    ] {
        let uy = month.to_underwriting_basis(Grain::Month, &basis).unwrap();
        // `from_annual` writes each month's premium at its start; written
        // at an even rate through the month, half of it is written as if
        // at the start of the next month, so its written increments are
        // the shares spread half and half over two months.
        let total: f64 = basis.writing.iter().sum();
        let n = 120;
        let mut written = vec![0.0; n];
        let mut cum = 0.0;
        for (t, slot) in written.iter_mut().enumerate() {
            let share = |j: usize| basis.writing.get(j).copied().unwrap_or(0.0) / total;
            cum += 0.5 * share(t) + if t > 0 { 0.5 * share(t - 1) } else { 0.0 };
            *slot = cum;
        }
        let engine = MonthlyPatterns::from_annual(
            written,
            basis.policy_months as usize,
            &paid,
            Interpolation::Linear,
        )
        .unwrap();
        for &(a, share) in &uy.values {
            assert_close(share, engine.paid[a as usize - 1], 1e-13, "from_annual");
        }
    }
}

// C. A simulated book.

/// The first month of the book, and the valuation.
const BOOK: i64 = 0; // January 2000
const START: i32 = 2000;
const VALUATION: i64 = 239; // December 2019

/// An accident month's share reported at age `k`: `1 - 0.9^k` to 60
/// months, held there.
fn report(k: i64) -> f64 {
    if k <= 0 {
        0.0
    } else {
        1.0 - 0.9f64.powi(k.min(60) as i32)
    }
}

/// A book of policies written every month from January 2000 to December
/// 2019, three a month at even times through it, the premium of each
/// `writing(month of the year)`, each earning evenly over `term` months.
/// Returns each policy's underwriting year and its exposure by accident
/// month (months since January 2000), losses at a rate of 1.
fn book(writing: &dyn Fn(usize) -> f64, term: u32) -> Vec<(i32, Vec<(i64, f64)>)> {
    let m = f64::from(term);
    let mut policies = Vec::new();
    for month in BOOK..=VALUATION {
        for i in 0..3 {
            let u = month as f64 + (f64::from(i) + 0.5) / 3.0;
            let premium = writing(month as usize % 12) / 3.0;
            let exposure = (month..=month + i64::from(term))
                .map(|t| {
                    let overlap = (t as f64 + 1.0).min(u + m) - (t as f64).max(u);
                    (t, premium * overlap.max(0.0) / m)
                })
                .collect();
            policies.push((START + (month / 12) as i32, exposure));
        }
    }
    policies
}

/// A triangle of origins from 2010 by year, valued every `grain` months to
/// December 2019; `cell(origin's first month, valuation month)` is its
/// cumulative value.
fn triangle(grain: Grain, cell: &dyn Fn(i64, i64) -> f64) -> Triangle {
    let g = i64::from(grain.months());
    let (mut origin, mut age, mut values) = (Vec::new(), Vec::new(), Vec::new());
    for y in 2010..=2019 {
        let first = i64::from(y - START) * 12;
        let mut a = g;
        while first + a - 1 <= VALUATION {
            origin.push(Month::january(y));
            age.push(a as Lag);
            values.push(cell(first, first + a - 1));
            a += g;
        }
    }
    Triangle::from_long(&Long {
        keys: &[],
        origin: &origin,
        development: DevelopmentColumn::Age(&age),
        values: &[("values", &values)],
        origin_grain: Grain::Year,
        development_grain: grain,
        cumulative: true,
    })
    .unwrap()
}

/// The book's accident-year and underwriting-year triangles, valued every
/// `grain` months.
fn triangles(policies: &[(i32, Vec<(i64, f64)>)], grain: Grain) -> (Triangle, Triangle) {
    let mut losses = vec![0.0; VALUATION as usize + 40];
    for (_, exposure) in policies {
        for &(t, x) in exposure {
            losses[t as usize] += x;
        }
    }
    let ay = triangle(grain, &|first, valuation| {
        (first..first + 12)
            .filter(|&t| t <= valuation)
            .map(|t| losses[t as usize] * report(valuation - t + 1))
            .sum()
    });
    let uy = triangle(grain, &|first, valuation| {
        let year = START + (first / 12) as i32;
        policies
            .iter()
            .filter(|p| p.0 == year)
            .flat_map(|p| &p.1)
            .map(|&(t, x)| x * report(valuation - t + 1))
            .sum()
    });
    (ay, uy)
}

/// The chain ladder pattern of `tri`, as percents developed at its ages.
fn pattern_of(fit: &ChainLadderFit) -> SelectedPattern {
    percent(
        fit.development
            .development
            .iter()
            .zip(&fit.cdf)
            .map(|(&a, &c)| (a, 1.0 / c))
            .collect(),
    )
}

#[test]
fn a_simulated_books_accident_year_pattern_converts_to_its_underwriting_year_pattern() {
    for term in [12, 6] {
        let basis = UnderwritingBasis::even(12, term);
        let policies = book(&|_| 1.0, term);
        let (ay, uy) = triangles(&policies, Grain::Month);
        let ay_fit = ChainLadder::default().fit(&ay, "values").unwrap();
        let uy_fit = ChainLadder::default().fit(&uy, "values").unwrap();
        // The book's accident years are fully developed at 71 months and
        // its underwriting years at 60 + 12 + term - 1, both well before
        // the oldest origin's 120 months.
        let converted = pattern_of(&ay_fit)
            .to_underwriting_basis(Grain::Year, &basis)
            .unwrap();
        for (&a, &c) in uy_fit.development.development.iter().zip(&uy_fit.cdf) {
            let p = if a <= converted.values.last().unwrap().0 {
                at(&converted, a)
            } else {
                1.0
            };
            assert_close(1.0 / c, p, 1e-12, &format!("term {term}, age {a}"));
        }
        // The converted pattern projects each underwriting year to its
        // ultimate, the losses on its exposure reported fully.
        let fit = chain_ladder(converted).fit(&uy, "values").unwrap();
        for (k, &ultimate) in fit.ultimate.iter().enumerate() {
            let year = 2010 + k as i32;
            let truth: f64 = policies
                .iter()
                .filter(|p| p.0 == year)
                .flat_map(|p| &p.1)
                .map(|&(_, x)| x * report(60))
                .sum();
            assert_close(ultimate, truth, 1e-12, &format!("ultimate {year}"));
        }
    }
}

#[test]
fn a_simulated_books_accident_month_pattern_converts_under_seasonal_writing() {
    let shares = [3.0, 1.0, 1.0, 2.0, 1.0, 4.0, 1.0, 1.0, 2.0, 1.0, 1.0, 6.0];
    let month = percent((1..=60).map(|k| (k, report(i64::from(k)))).collect());
    for term in [12, 6, 18] {
        let policies = book(&|j| shares[j], term);
        let (_, uy) = triangles(&policies, Grain::Month);
        let uy_fit = ChainLadder::default().fit(&uy, "values").unwrap();
        let converted = month
            .to_underwriting_basis(Grain::Month, &UnderwritingBasis::new(shares.to_vec(), term))
            .unwrap();
        let level = report(60);
        for (&a, &c) in uy_fit.development.development.iter().zip(&uy_fit.cdf) {
            let p = if a <= converted.values.last().unwrap().0 {
                at(&converted, a)
            } else {
                level
            };
            assert_close(level / c, p, 1e-12, &format!("term {term}, age {a}"));
        }
    }
}

/// The misses of the underwriting year's factors to ultimate when the
/// book's accident-year triangle valued annually is converted under
/// `basis`: `cdf * p - 1` at each of its ages.
fn annual_misses(basis: &UnderwritingBasis) -> Vec<(Lag, f64)> {
    let shares = basis.writing.clone();
    let policies = book(&|j| shares[j], basis.policy_months);
    let (ay, uy) = triangles(&policies, Grain::Year);
    let ay_fit = ChainLadder::default().fit(&ay, "values").unwrap();
    let uy_fit = ChainLadder::default().fit(&uy, "values").unwrap();
    let converted = pattern_of(&ay_fit)
        .to_underwriting_basis(Grain::Year, basis)
        .unwrap();
    let mut misses = Vec::new();
    for (&a, &c) in uy_fit.development.development.iter().zip(&uy_fit.cdf) {
        let p = if a <= converted.values.last().unwrap().0 {
            at(&converted, a)
        } else {
            1.0
        };
        misses.push((a, c * p - 1.0));
    }
    misses
}

#[test]
fn an_annual_accident_year_pattern_converts_approximately_under_decision_18() {
    // Between annual ages the accident-year pattern read ln-linearly is not
    // the even-exposure average of its months: the miss on the underwriting
    // year's factors to ultimate, pinned.
    let misses = annual_misses(&UnderwritingBasis::default().with_reading(AnnualReading::LnLinear));
    // The converted share over the book's own: 31% too high at 12 months,
    // where the accident-year pattern is read linearly from 0 (decision 18
    // item 2), 8% too low at 24, within 0.1% from 60 months, and exact
    // once every accident year is past its end.
    let pinned = [
        (12, 0.311_566_534),
        (24, -0.084_495_709),
        (36, -0.013_362_415),
        (48, -0.003_300_967),
        (60, -0.000_899_254),
        (72, -0.000_350_028),
    ];
    for (a, miss) in pinned {
        let found = misses.iter().find(|m| m.0 == a).unwrap().1;
        assert!((found - miss).abs() < 1e-8, "age {a}: {found}");
    }
    assert!(
        misses
            .iter()
            .filter(|m| m.0 >= 84)
            .all(|m| m.1.abs() < 1e-14)
    );
}

#[test]
fn an_annual_accident_year_pattern_converts_closely_under_the_default_reading() {
    // The Weibull fitted through the yearly averages and corrected to
    // reproduce them (decision 25): within 0.2% of the book's own
    // underwriting year at every age.
    assert_eq!(
        UnderwritingBasis::default().reading(),
        AnnualReading::Curve(GrowthCurve::Weibull)
    );
    let misses = annual_misses(&UnderwritingBasis::default());
    // The miss of the converted share at each age, pinned: 0.11% low at
    // 12 months, within 0.04% from 24 and exact once every accident year
    // is past its end.
    let pinned = [
        (12, -0.001_072_193_817),
        (24, 0.000_373_870_029),
        (36, -0.000_095_800_352),
        (48, 0.000_016_862_671),
        (60, 0.000_015_564_668),
        (72, -0.000_015_215_385),
    ];
    for (a, miss) in pinned {
        let found = misses.iter().find(|m| m.0 == a).unwrap().1;
        assert!((found - miss).abs() < 1e-8, "age {a}: {found}");
    }
    // The study's ratios to four digits.
    for (a, ratio) in [
        (12, 0.9989),
        (24, 1.0004),
        (36, 0.9999),
        (48, 1.0),
        (60, 1.0),
    ] {
        let found = misses.iter().find(|m| m.0 == a).unwrap().1;
        assert!((1.0 + found - ratio).abs() < 1e-4, "age {a}: {found}");
    }
    assert!(misses.iter().all(|m| m.1.abs() < 2e-3));
    // Seasonal writing (three times the rate in the last quarter) and
    // six-month policies: the study's ratios at 12 months, where decision
    // 18's reading is 1.40 and 1.17.
    let seasonal = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 3.0, 3.0, 3.0];
    for (basis, curve, ln_linear) in [
        (UnderwritingBasis::new(seasonal.to_vec(), 12), 0.9986, 1.40),
        (UnderwritingBasis::even(12, 6), 0.9994, 1.17),
    ] {
        let old = annual_misses(&basis.clone().with_reading(AnnualReading::LnLinear));
        let new = annual_misses(&basis);
        assert!(
            (1.0 + old[0].1 - ln_linear).abs() < 5e-3,
            "{basis:?}: {}",
            old[0].1
        );
        assert!(
            (1.0 + new[0].1 - curve).abs() < 1e-4,
            "{basis:?}: {}",
            new[0].1
        );
        assert!(new.iter().all(|m| m.1.abs() < 2e-3), "{basis:?}");
    }
    assert!(
        misses
            .iter()
            .filter(|m| m.0 >= 84)
            .all(|m| m.1.abs() < 1e-14)
    );
}

// D. Against the design study's scipy prototype.

/// The book's accident-year pattern at 12 to 120 months: the twelve-month
/// averages of `1 - 0.9^t`, held from 60.
fn book_annual() -> SelectedPattern {
    percent(
        (1..=10)
            .map(|k| {
                let a = 12 * k;
                let sum: f64 = (0..12).map(|q| report(i64::from(a - q))).sum();
                (a, sum / 12.0)
            })
            .collect(),
    )
}

/// CLRD private passenger auto paid, the volume-weighted chain ladder of
/// every company summed, as percents developed at 12 to 120 months
/// (`industry_patterns()` of the study's script, from
/// `validation/data/clrd_lines.csv`).
fn ppauto() -> SelectedPattern {
    percent(vec![
        (12, 0.390_734_934_936_675_6),
        (24, 0.705_876_740_467_845_8),
        (36, 0.846_998_041_767_789_9),
        (48, 0.922_266_658_057_296_9),
        (60, 0.961_798_515_109_469_7),
        (72, 0.981_468_825_098_16),
        (84, 0.991_327_739_424_762_4),
        (96, 0.996_416_276_351_787_9),
        (108, 0.999_127_161_268_376_8),
        (120, 1.0),
    ])
}

#[test]
fn the_default_reading_matches_the_scipy_prototype() {
    // The converted underwriting-year share at 12 to 72 months under the
    // default parallelogram, from the study's `expand_curve_exact` then
    // `convert` (scipy Nelder-Mead from a grid of starts for the curve,
    // SLSQP at `ftol = 1e-14` for the projection).
    let cases = [
        (
            "book",
            book_annual(),
            GrowthCurve::Weibull,
            [
                0.175_868_694_912,
                0.694_533_317_249,
                0.913_566_348_240,
                0.975_629_755_857,
                0.993_127_934_330,
                0.997_723_201_987,
            ],
        ),
        (
            "book",
            book_annual(),
            GrowthCurve::LogLogistic,
            [
                0.163_876_635_990,
                0.710_262_202_810,
                0.908_728_555_544,
                0.976_997_156_960,
                0.992_801_254_096,
                0.997_783_447_176,
            ],
        ),
        (
            "ppauto",
            ppauto(),
            GrowthCurve::Weibull,
            [
                0.153_188_741_398,
                0.574_839_269_322,
                0.783_759_840_977,
                0.888_517_415_103,
                0.944_202_929_534,
                0.972_787_000_939,
            ],
        ),
        (
            "ppauto",
            ppauto(),
            GrowthCurve::LogLogistic,
            [
                0.146_128_465_060,
                0.584_185_829_055,
                0.780_856_690_628,
                0.889_287_085_058,
                0.944_003_057_713,
                0.972_842_852_533,
            ],
        ),
    ];
    // The two agree within 1.1e-9 (the loglogistic on the book; the
    // Weibull within 5e-10): both curve searches stop near 1e-12 of their
    // tolerance, not at the same point, and SLSQP's projection is exact
    // to about 1e-14.
    for (name, pattern, curve, scipy) in cases {
        let basis = UnderwritingBasis::default().with_reading(AnnualReading::Curve(curve));
        let uy = pattern.to_underwriting_basis(Grain::Year, &basis).unwrap();
        for (k, want) in scipy.into_iter().enumerate() {
            let a = 12 * (k as Lag + 1);
            let found = at(&uy, a);
            assert!(
                (found - want).abs() < 1e-8,
                "{name} {curve:?} at {a}: {found} against {want}"
            );
        }
    }
}
