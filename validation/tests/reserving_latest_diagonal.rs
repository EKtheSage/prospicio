//! A latest diagonal only, projected with a fully selected pattern
//! (decision 15 of `docs/design/reserving-v02.md`, issue #544).
//!
//! * Hand calculation: a six-origin diagonal with a pattern to ultimate,
//!   through the chain ladder (`latest × cdf`), Bornhuetter–Ferguson
//!   (`latest + ELR × premium × (1 - 1/cdf)`), Benktander (two iterations
//!   of the credibility formula) and Cape Cod (`Σ latest / Σ premium ×
//!   %reported`), and the expected loss method and percents developed.
//! * Equivalence: the latest diagonal cut from GenIns and RAA, with the
//!   full triangle's own factors selected as age-to-age factors or factors
//!   to ultimate, gives every origin the full triangle's chain ladder
//!   ultimate (with each estimated tail too, on GenIns); the diagonal cut
//!   from the CAS workers' compensation triangle gives its
//!   Bornhuetter-Ferguson, Benktander and Cape Cod ultimates.
//! * Errors: a pattern that leaves an age to estimate, or a fully selected
//!   age emptied by `n_periods`, keeps its error; a pattern age before the
//!   youngest origin is off the grid; Mack, the ODP bootstrap and Mack's
//!   bootstrap name the latest diagonal, and not a triangle with holes.
//!
//! The dedicated type, `LatestDiagonal` (decision 16):
//!
//! * the same fits as the triangle path, exactly, for every method and
//!   pattern kind on the diagonals above;
//! * chainladder-python parity: `DevelopmentConstant` on the latest
//!   diagonal with the chain ladder on GenIns and the expected-loss methods
//!   on the CAS workers' compensation triangle
//!   (`reference/reserving_latest_diagonal_python.csv`, from
//!   `scripts/reserving_latest_diagonal_python.py`);
//! * ages off the pattern's ages, on half-year, quarterly and monthly
//!   grids, by the log-linear rule, and age-to-age factors on a finer grid;
//! * an estimated tail fitted on the pattern's own ages, an origin at the
//!   factors' end taking the tail alone, ages from a valuation, as a
//!   triangle's, and the errors of the type and its pattern.

use std::collections::HashMap;

use prospicio_reserving::{
    Benktander, BornhuetterFerguson, CapeCod, ChainLadder, Development, DevelopmentColumn, Error,
    ExpectedLoss, Grain, Lag, LatestDiagonal, Long, Mack, MackBootstrap, Month, OdpBootstrap,
    PastPatternEnd, PatternKind, PatternWeight, Period, SelectedPattern, Tail, Triangle,
};
use prospicio_validation::{Case, check, reference, triangle, triangle_columns};

/// The latest observed cell of every origin of a single-segment `tri`, with
/// every column's value there, as a triangle of its own.
fn latest_diagonal(tri: &Triangle) -> Triangle {
    let ages = tri.development();
    let starts: Vec<Month> = tri.origins().iter().map(|p| p.start()).collect();
    let columns = tri.columns().to_vec();
    let mut origin = Vec::new();
    let mut age = Vec::new();
    let mut values = vec![Vec::new(); columns.len()];
    for (o, &start) in starts.iter().enumerate() {
        let d = (0..ages.len())
            .rev()
            .find(|&d| tri.get(0, 0, o, d).is_some())
            .expect("an origin with no observed value");
        origin.push(start);
        age.push(ages[d]);
        for (c, column) in values.iter_mut().enumerate() {
            column.push(tri.get(0, c, o, d).expect("a missing cell on the diagonal"));
        }
    }
    let values: Vec<(&str, &[f64])> = columns
        .iter()
        .zip(&values)
        .map(|(n, v)| (n.as_str(), v.as_slice()))
        .collect();
    Triangle::from_long(&Long {
        keys: &[],
        origin: &origin,
        development: DevelopmentColumn::Age(&age),
        values: &values,
        origin_grain: tri.origin_grain(),
        development_grain: tri.development_grain(),
        cumulative: true,
    })
    .unwrap()
}

/// The chain ladder with `pattern` fully selected.
fn selected(pattern: SelectedPattern) -> ChainLadder {
    ChainLadder {
        development: Development {
            pattern: Some(pattern),
            ..Default::default()
        },
        ..Default::default()
    }
}

/// The triangle's own age-to-age factors and factors to ultimate, by age.
fn own_patterns(tri: &Triangle, column: &str) -> [SelectedPattern; 2] {
    let fit = ChainLadder::default().fit(tri, column).unwrap();
    let ages = tri.development();
    let ldf = fit.ldf().to_vec();
    let mut cdf = vec![1.0; ages.len()];
    for k in (0..ldf.len()).rev() {
        cdf[k] = cdf[k + 1] * ldf[k];
    }
    [
        SelectedPattern::new(
            PatternKind::AgeToAge,
            ages.iter().copied().zip(ldf).collect(),
        ),
        SelectedPattern::new(
            PatternKind::AgeToUltimate,
            ages.iter().copied().zip(cdf).collect(),
        ),
    ]
}

fn assert_close(found: &[f64], expected: &[f64], what: &str) {
    assert_eq!(found.len(), expected.len(), "{what}: lengths");
    for (o, (f, e)) in found.iter().zip(expected).enumerate() {
        assert!(
            (f - e).abs() <= 1e-9 * e.abs().max(1.0),
            "{what}, origin {o}: {f} against {e}"
        );
    }
}

/// Six accident years at 31 December 2024, each at its own age.
struct Hand {
    tri: Triangle,
    latest: [f64; 6],
    premium: f64,
    cdf: [f64; 6],
}

fn hand() -> Hand {
    let latest = [980.0, 860.0, 700.0, 520.0, 400.0, 150.0];
    let premium = 200.0;
    // To ultimate at 72, 60, ..., 12 months, so the oldest origin first.
    let cdf = [1.0, 1.02, 1.07, 1.15, 1.3, 12.0 / 7.0];
    let origin: Vec<Month> = (2019..=2024).map(Month::january).collect();
    let age: Vec<Lag> = vec![72, 60, 48, 36, 24, 12];
    let premiums = [premium; 6];
    let tri = Triangle::from_long(&Long {
        keys: &[],
        origin: &origin,
        development: DevelopmentColumn::Age(&age),
        values: &[("paid", &latest[..]), ("premium", &premiums[..])],
        origin_grain: Grain::Year,
        development_grain: Grain::Year,
        cumulative: true,
    })
    .unwrap();
    Hand {
        tri,
        latest,
        premium,
        cdf,
    }
}

fn hand_pattern(h: &Hand) -> SelectedPattern {
    let ages: [Lag; 6] = [12, 24, 36, 48, 60, 72];
    let by_age: Vec<(Lag, f64)> = ages
        .iter()
        .zip(h.cdf.iter().rev())
        .map(|(&a, &c)| (a, c))
        .collect();
    SelectedPattern::new(PatternKind::AgeToUltimate, by_age)
}

#[test]
fn a_diagonal_is_projected_by_hand() {
    let h = hand();
    let chain_ladder = selected(hand_pattern(&h));
    let elr = 0.7;
    let reported: Vec<f64> = h.cdf.iter().map(|c| 1.0 / c).collect();

    let cl = chain_ladder.fit(&h.tri, "paid").unwrap();
    let expected: Vec<f64> = h.latest.iter().zip(&h.cdf).map(|(l, c)| l * c).collect();
    assert_close(&cl.ultimate, &expected, "chain ladder");
    assert!((cl.ultimate[5] - 257.142_857_142_857).abs() < 1e-9);
    assert!(cl.development.n_links.iter().all(|&n| n == 0));
    assert!(cl.development.has_missing_factor());

    let bf = BornhuetterFerguson {
        apriori: elr.into(),
        chain_ladder: chain_ladder.clone(),
    }
    .fit(&h.tri, "paid", "premium")
    .unwrap();
    let expected: Vec<f64> = h
        .latest
        .iter()
        .zip(&reported)
        .map(|(l, p)| l + elr * h.premium * (1.0 - p))
        .collect();
    assert_close(&bf.ultimate, &expected, "Bornhuetter-Ferguson");
    assert!((bf.ultimate[5] - 208.333_333_333_333).abs() < 1e-9);

    let bk = Benktander {
        apriori: elr.into(),
        n_iters: 2,
        chain_ladder: chain_ladder.clone(),
    }
    .fit(&h.tri, "paid", "premium")
    .unwrap();
    let expected: Vec<f64> = h
        .latest
        .iter()
        .zip(&reported)
        .map(|(l, p)| {
            let bf = l + elr * h.premium * (1.0 - p);
            l + bf * (1.0 - p)
        })
        .collect();
    assert_close(&bk.ultimate, &expected, "Benktander");

    let cc = CapeCod {
        trend: 0.0,
        decay: 1.0,
        chain_ladder,
    }
    .fit(&h.tri, "paid", "premium")
    .unwrap();
    let used: f64 = reported.iter().map(|p| h.premium * p).sum();
    let cc_elr = h.latest.iter().sum::<f64>() / used;
    let expected: Vec<f64> = h
        .latest
        .iter()
        .zip(&reported)
        .map(|(l, p)| l + cc_elr * h.premium * (1.0 - p))
        .collect();
    assert_close(&cc.expected_loss.ultimate, &expected, "Cape Cod");
}

#[test]
fn a_diagonal_takes_percents_developed_and_the_expected_loss_method() {
    let h = hand();
    let ages: [Lag; 6] = [12, 24, 36, 48, 60, 72];
    let percents: Vec<(Lag, f64)> = ages
        .iter()
        .zip(h.cdf.iter().rev())
        .map(|(&a, &c)| (a, 1.0 / c))
        .collect();
    let chain_ladder = selected(SelectedPattern::new(
        PatternKind::PercentDeveloped,
        percents,
    ));
    let cl = chain_ladder.fit(&h.tri, "paid").unwrap();
    let expected: Vec<f64> = h.latest.iter().zip(&h.cdf).map(|(l, c)| l * c).collect();
    assert_close(
        &cl.ultimate,
        &expected,
        "chain ladder on percents developed",
    );

    let el = ExpectedLoss {
        apriori: 0.7.into(),
        chain_ladder,
    }
    .fit(&h.tri, "paid", "premium")
    .unwrap();
    assert_close(&el.ultimate, &[0.7 * h.premium; 6], "expected loss");
}

#[test]
fn a_diagonal_cut_from_a_triangle_keeps_its_chain_ladder_ultimates() {
    for name in ["genins", "raa"] {
        let full = triangle(name);
        let diagonal = latest_diagonal(&full);
        let expected = ChainLadder::default()
            .fit(&full, "values")
            .unwrap()
            .ultimate;
        for pattern in own_patterns(&full, "values") {
            let kind = pattern.kind;
            let found = selected(pattern).fit(&diagonal, "values").unwrap().ultimate;
            assert_close(&found, &expected, &format!("{name} {kind:?}"));
        }
    }
}

#[test]
fn a_diagonal_cut_from_a_triangle_keeps_its_expected_loss_ultimates() {
    let full = triangle_columns("clrd_wkcomp", &["paid", "premium"]);
    let diagonal = latest_diagonal(&full);
    for pattern in own_patterns(&full, "paid") {
        let kind = pattern.kind;
        let chain_ladder = selected(pattern);
        let both = |f: &dyn Fn(&Triangle) -> Vec<f64>, what: &str| {
            assert_close(&f(&diagonal), &f(&full), &format!("{what} {kind:?}"));
        };
        both(
            &|t| {
                BornhuetterFerguson {
                    apriori: 0.6.into(),
                    chain_ladder: chain_ladder.clone(),
                }
                .fit(t, "paid", "premium")
                .unwrap()
                .ultimate
            },
            "Bornhuetter-Ferguson",
        );
        both(
            &|t| {
                Benktander {
                    apriori: 0.6.into(),
                    n_iters: 3,
                    chain_ladder: chain_ladder.clone(),
                }
                .fit(t, "paid", "premium")
                .unwrap()
                .ultimate
            },
            "Benktander",
        );
        both(
            &|t| {
                CapeCod {
                    trend: 0.02,
                    decay: 0.75,
                    chain_ladder: chain_ladder.clone(),
                }
                .fit(t, "paid", "premium")
                .unwrap()
                .expected_loss
                .ultimate
            },
            "Cape Cod",
        );
    }
}

#[test]
fn an_age_left_to_estimate_keeps_the_link_ratio_error() {
    let h = hand();
    // The pattern stops at 48 months: the factors from 48 on are the
    // triangle's own, which a diagonal cannot estimate.
    let partial = SelectedPattern::new(
        PatternKind::AgeToAge,
        vec![(12, 1.3), (24, 1.15), (36, 1.07)],
    );
    let e = selected(partial).fit(&h.tri, "paid").unwrap_err();
    assert_eq!(
        e,
        Error::Factor {
            age: 3,
            reason: "no origin is observed at both ages"
        }
    );
    // And with no pattern, at the first age.
    let e = ChainLadder::default().fit(&h.tri, "paid").unwrap_err();
    assert_eq!(
        e.to_string(),
        "factor from development index 0: no origin is observed at both ages"
    );
}

#[test]
fn the_methods_that_need_link_ratios_name_the_latest_diagonal() {
    let h = hand();
    let development = Development {
        pattern: Some(hand_pattern(&h)),
        ..Default::default()
    };
    let mack = Mack {
        development: development.clone(),
        ..Default::default()
    };
    assert_eq!(
        mack.fit(&h.tri, "paid").unwrap_err(),
        Error::LatestDiagonalOnly {
            method: "Mack's model"
        }
    );
    let odp = OdpBootstrap {
        pattern: Some(hand_pattern(&h)),
        ..Default::default()
    };
    assert_eq!(
        odp.fit(&h.tri, "paid").unwrap_err(),
        Error::LatestDiagonalOnly {
            method: "the ODP bootstrap"
        }
    );
    let mack_boot = MackBootstrap {
        development,
        ..Default::default()
    };
    assert_eq!(
        mack_boot.fit(&h.tri, "paid").unwrap_err(),
        Error::LatestDiagonalOnly {
            method: "Mack's bootstrap"
        }
    );
    let message = Error::LatestDiagonalOnly {
        method: "Mack's model",
    }
    .to_string();
    assert!(message.contains("latest diagonal"), "{message}");
}

#[test]
fn a_given_tail_applies_to_a_diagonal() {
    let h = hand();
    let ages: [Lag; 6] = [12, 24, 36, 48, 60, 72];
    let ldf = [1.3, 1.15, 1.07, 1.02, 1.01];
    let pattern = SelectedPattern::new(
        PatternKind::AgeToAge,
        ages.iter().copied().zip(ldf).collect(),
    );
    let chain_ladder = ChainLadder {
        tail: Tail::from(1.05),
        ..selected(pattern)
    };
    let fit = chain_ladder.fit(&h.tri, "paid").unwrap();
    // The oldest origin, at 72 months, takes the tail alone.
    assert!((fit.ultimate[0] - h.latest[0] * 1.05).abs() < 1e-9);
    let youngest: f64 = ldf.iter().product::<f64>() * 1.05;
    assert!((fit.ultimate[5] - h.latest[5] * youngest).abs() < 1e-9);
}

/// An estimated tail is fitted to the factors in use, the selected ones
/// here, so a diagonal takes the full triangle's tail.
#[test]
fn an_estimated_tail_is_fitted_to_the_selected_factors() {
    let full = triangle("genins");
    let diagonal = latest_diagonal(&full);
    let [ldf, _] = own_patterns(&full, "values");
    for tail in [
        Tail::LogLinear,
        Tail::Curve(Default::default()),
        Tail::Bondy(Default::default()),
    ] {
        let chain_ladder = ChainLadder {
            tail,
            ..selected(ldf.clone())
        };
        let expected = ChainLadder {
            tail,
            ..Default::default()
        }
        .fit(&full, "values")
        .unwrap();
        let found = chain_ladder.fit(&diagonal, "values").unwrap();
        assert!(expected.tail.factor > 1.0, "{tail:?}");
        assert!(
            (found.tail.factor - expected.tail.factor).abs() < 1e-12,
            "{tail:?}"
        );
        assert_close(&found.ultimate, &expected.ultimate, &format!("{tail:?}"));
    }
}

/// A triangle from rows of `(origin year, age, value)`.
fn rows(cells: &[(i32, Lag, f64)]) -> Triangle {
    let origin: Vec<Month> = cells.iter().map(|c| Month::january(c.0)).collect();
    let age: Vec<Lag> = cells.iter().map(|c| c.1).collect();
    let values: Vec<f64> = cells.iter().map(|c| c.2).collect();
    Triangle::from_long(&Long {
        keys: &[],
        origin: &origin,
        development: DevelopmentColumn::Age(&age),
        values: &[("paid", &values)],
        origin_grain: Grain::Year,
        development_grain: Grain::Year,
        cumulative: true,
    })
    .unwrap()
}

fn full_pattern() -> SelectedPattern {
    SelectedPattern::new(PatternKind::AgeToAge, vec![(12, 1.5), (24, 1.2)])
}

#[test]
fn a_triangle_with_holes_is_not_a_latest_diagonal() {
    // 2020 at 12 and 36 months, a hole at 24; 2021 at 24 only; 2022 at 12.
    // No origin is observed at two consecutive ages, but 2020 is observed
    // twice, so the errors are the ones the holes always gave.
    let tri = rows(&[
        (2020, 12, 100.0),
        (2020, 36, 180.0),
        (2021, 24, 150.0),
        (2022, 12, 90.0),
    ]);
    let development = Development {
        pattern: Some(full_pattern()),
        ..Default::default()
    };
    let errors = [
        Mack {
            development: development.clone(),
            ..Default::default()
        }
        .fit(&tri, "paid")
        .unwrap_err(),
        MackBootstrap {
            development,
            ..Default::default()
        }
        .fit(&tri, "paid")
        .unwrap_err(),
        OdpBootstrap {
            pattern: Some(full_pattern()),
            ..Default::default()
        }
        .fit(&tri, "paid")
        .unwrap_err(),
    ];
    for e in errors {
        assert!(
            !matches!(e, Error::LatestDiagonalOnly { .. } | Error::Pattern { .. }),
            "{e}"
        );
    }
}

#[test]
fn a_fully_selected_age_emptied_by_n_periods_keeps_its_error() {
    // 2021 lags at 12 months, so the only link from 12 months, 2020's,
    // starts before the latest period that n_periods = 1 keeps.
    let tri = rows(&[
        (2020, 12, 100.0),
        (2020, 24, 150.0),
        (2020, 36, 165.0),
        (2021, 12, 90.0),
        (2022, 12, 95.0),
    ]);
    let chain_ladder = ChainLadder {
        development: Development {
            pattern: Some(full_pattern()),
            n_periods: Some(1),
            ..Default::default()
        },
        ..Default::default()
    };
    assert_eq!(
        chain_ladder.fit(&tri, "paid").unwrap_err(),
        Error::Factor {
            age: 0,
            reason: "no link ratio is left after n_periods"
        }
    );
}

#[test]
fn a_pattern_age_before_the_youngest_origin_is_read() {
    // The youngest origin is at 24 months, so the triangle's ages start
    // there; the pattern's age of 12 months is off them, and the pattern is
    // read on its own ages (decision 18), as the diagonal type reads it.
    let h = hand();
    let tri = rows(&[
        (2019, 72, h.latest[0]),
        (2020, 60, h.latest[1]),
        (2021, 48, h.latest[2]),
        (2022, 36, h.latest[3]),
        (2023, 24, h.latest[4]),
    ]);
    let fit = selected(hand_pattern(&h)).fit(&tri, "paid").unwrap();
    let want: Vec<f64> = (0..5).map(|o| h.latest[o] * h.cdf[o]).collect();
    assert_close(&fit.ultimate, &want, "the triangle path");
    let origins = (2019..=2023).map(Month::january).collect::<Vec<_>>();
    let d =
        LatestDiagonal::new(Grain::Year, &origins, &[72, 60, 48, 36, 24], &h.latest[..5]).unwrap();
    let diagonal = selected(hand_pattern(&h)).fit_diagonal(&d).unwrap();
    for (f, e) in fit.ultimate.iter().zip(&diagonal.ultimate) {
        assert!((f - e).abs() <= 1e-14 * e.abs(), "{f} against {e}");
    }
    assert_eq!(fit.pattern_tail, Some(1.0));
}

// The dedicated type (decision 16).

/// The type's diagonal of a single-segment `tri`, with `exposure` from that
/// column when given.
fn diagonal_type(tri: &Triangle, column: &str, exposure: Option<&str>) -> LatestDiagonal {
    let ages = tri.development();
    let c = tri.column_position(column).unwrap();
    let mut origins = Vec::new();
    let mut at = Vec::new();
    let mut latest = Vec::new();
    let mut premium = Vec::new();
    for (o, p) in tri.origins().iter().enumerate() {
        let d = (0..ages.len())
            .rev()
            .find(|&d| tri.get(0, c, o, d).is_some())
            .unwrap();
        origins.push(p.start());
        at.push(ages[d]);
        latest.push(tri.get(0, c, o, d).unwrap());
        if let Some(e) = exposure {
            premium.push(tri.get(0, tri.column_position(e).unwrap(), o, d).unwrap());
        }
    }
    let d = LatestDiagonal::new(tri.origin_grain(), &origins, &at, &latest).unwrap();
    match exposure {
        Some(_) => d.with_exposure(&premium).unwrap(),
        None => d,
    }
}

#[test]
fn the_type_gives_exactly_the_triangle_path() {
    for name in ["genins", "raa"] {
        let full = triangle(name);
        let cut = latest_diagonal(&full);
        let d = diagonal_type(&full, "values", None);
        let [ldf, cdf] = own_patterns(&full, "values");
        let percent = SelectedPattern::new(
            PatternKind::PercentDeveloped,
            cdf.values.iter().map(|&(a, c)| (a, 1.0 / c)).collect(),
        );
        for pattern in [ldf, cdf, percent] {
            let kind = pattern.kind;
            let cl = selected(pattern);
            assert_eq!(
                cl.fit_diagonal(&d).unwrap().ultimate,
                cl.fit(&cut, "values").unwrap().ultimate,
                "{name} {kind:?}"
            );
        }
    }
    let full = triangle_columns("clrd_wkcomp", &["paid", "premium"]);
    let cut = latest_diagonal(&full);
    let d = diagonal_type(&full, "paid", Some("premium"));
    let [ldf, cdf] = own_patterns(&full, "paid");
    let percent = SelectedPattern::new(
        PatternKind::PercentDeveloped,
        cdf.values.iter().map(|&(a, c)| (a, 1.0 / c)).collect(),
    );
    for pattern in [ldf, cdf, percent] {
        let chain_ladder = selected(pattern);
        let el = ExpectedLoss {
            apriori: 0.6.into(),
            chain_ladder: chain_ladder.clone(),
        };
        assert_eq!(
            el.fit_diagonal(&d).unwrap().ultimate,
            el.fit(&cut, "paid", "premium").unwrap().ultimate
        );
        let bf = BornhuetterFerguson {
            apriori: 0.6.into(),
            chain_ladder: chain_ladder.clone(),
        };
        assert_eq!(
            bf.fit_diagonal(&d).unwrap().ultimate,
            bf.fit(&cut, "paid", "premium").unwrap().ultimate
        );
        let bk = Benktander {
            apriori: 0.6.into(),
            n_iters: 3,
            chain_ladder: chain_ladder.clone(),
        };
        assert_eq!(
            bk.fit_diagonal(&d).unwrap().ultimate,
            bk.fit(&cut, "paid", "premium").unwrap().ultimate
        );
        let cc = CapeCod {
            trend: 0.02,
            decay: 0.75,
            chain_ladder,
        };
        let (a, b) = (
            cc.fit_diagonal(&d).unwrap(),
            cc.fit(&cut, "paid", "premium").unwrap(),
        );
        assert_eq!(a.expected_loss.ultimate, b.expected_loss.ultimate);
        assert_eq!(a.trended_apriori, b.trended_apriori);
    }
}

/// chainladder-python's `DevelopmentConstant` on the latest diagonal
/// (`reference/reserving_latest_diagonal_python.csv`, from
/// `scripts/reserving_latest_diagonal_python.py`). Its age-to-age pattern
/// carries a factor of 1 at the oldest age, which the type reads as a
/// factor from 120 to 132 months.
#[test]
fn the_type_matches_chainladder_python() {
    let cases = reference("reserving_latest_diagonal_python.csv");
    let mut patterns: HashMap<(String, String), Vec<(Lag, f64)>> = HashMap::new();
    for case in cases.iter().filter(|c| c.get("quantity") == "pattern") {
        patterns
            .entry((case.get("dataset").into(), case.get("method").into()))
            .or_default()
            .push((
                case.number("arg").unwrap() as Lag,
                case.number("expected").unwrap(),
            ));
    }
    let checked: Vec<Case> = cases
        .into_iter()
        .filter(|c| c.get("quantity") != "pattern")
        .collect();
    let mut fits: HashMap<(String, String), (Vec<Period>, Vec<f64>)> = HashMap::new();
    check(&checked, |case: &Case| {
        let (dataset, method) = (case.get("dataset"), case.get("method"));
        let key = (dataset.to_string(), method.to_string());
        let (origins, ultimate) = fits.entry(key.clone()).or_insert_with(|| {
            let kind = if method.contains("style=ldf") {
                PatternKind::AgeToAge
            } else {
                PatternKind::AgeToUltimate
            };
            let chain_ladder = selected(SelectedPattern::new(kind, patterns[&key].clone()));
            let param = |name| case.param("method", name);
            if dataset == "genins" {
                let full = triangle(dataset);
                let fit = chain_ladder
                    .fit_diagonal(&diagonal_type(&full, "values", None))
                    .unwrap();
                return (fit.origins, fit.ultimate);
            }
            let full = triangle_columns(dataset, &["paid", "premium"]);
            let d = diagonal_type(&full, "paid", Some("premium"));
            let fit = match method.split(';').next().unwrap() {
                "bornhuetter_ferguson" => BornhuetterFerguson {
                    apriori: param("apriori").into(),
                    chain_ladder,
                }
                .fit_diagonal(&d),
                "benktander" => Benktander {
                    apriori: param("apriori").into(),
                    n_iters: param("n_iters") as usize,
                    chain_ladder,
                }
                .fit_diagonal(&d),
                "cape_cod" => CapeCod {
                    trend: param("trend"),
                    decay: param("decay"),
                    chain_ladder,
                }
                .fit_diagonal(&d)
                .map(|f| f.expected_loss),
                other => panic!("unknown method {other}"),
            }
            .unwrap();
            (fit.chain_ladder.origins, fit.ultimate)
        });
        match case.get("quantity") {
            "total_ultimate" => Some(ultimate.iter().sum()),
            "ultimate" => {
                let year = case.number("arg")? as i32;
                let o = origins.iter().position(|p| *p == Period::year(year))?;
                Some(ultimate[o])
            }
            _ => None,
        }
    });
}

/// Factors to ultimate of 2, 1.25 and 1 at 12, 24 and 36 months.
fn annual_cdf() -> SelectedPattern {
    SelectedPattern::new(
        PatternKind::AgeToUltimate,
        vec![(12, 2.0), (24, 1.25), (36, 1.0)],
    )
}

#[test]
fn an_age_between_the_patterns_is_read_log_linearly() {
    // Half-year origins at December 2024: 24, 18 and 12 months.
    let origins = [
        Month::new(2023, 1).unwrap(),
        Month::new(2023, 7).unwrap(),
        Month::new(2024, 1).unwrap(),
    ];
    let d = LatestDiagonal::at_valuation(
        Grain::Semester,
        &origins,
        Month::new(2024, 12).unwrap(),
        &[500.0, 300.0, 100.0],
    )
    .unwrap()
    .with_exposure(&[800.0; 3])
    .unwrap();
    assert_eq!(d.ages(), [24, 18, 12]);
    let cdf18 = (2.0f64 * 1.25).sqrt();
    let cl = selected(annual_cdf()).fit_diagonal(&d).unwrap();
    assert_eq!(cl.development.development, [12, 18, 24, 30, 36]);
    assert_close(
        &cl.ultimate,
        &[500.0 * 1.25, 300.0 * cdf18, 100.0 * 2.0],
        "semester chain ladder",
    );
    let bf = BornhuetterFerguson {
        apriori: 0.5.into(),
        chain_ladder: selected(annual_cdf()),
    }
    .fit_diagonal(&d)
    .unwrap();
    assert_close(
        &bf.ultimate,
        &[
            500.0 + 400.0 * (1.0 - 1.0 / 1.25),
            300.0 + 400.0 * (1.0 - 1.0 / cdf18),
            100.0 + 400.0 * 0.5,
        ],
        "semester Bornhuetter-Ferguson",
    );

    // The same factors as age-to-age factors: 1.6 from 12 to 24 months and
    // 1.25 from 24 to 36.
    let ldf = SelectedPattern::new(PatternKind::AgeToAge, vec![(12, 1.6), (24, 1.25)]);
    let by_ldf = selected(ldf).fit_diagonal(&d).unwrap();
    assert_close(&by_ldf.ultimate, &cl.ultimate, "age-to-age on a finer grid");
    // And a tail past 36 months multiplies every origin's factor alike.
    let with_tail = ChainLadder {
        tail: Tail::from(1.05),
        ..selected(SelectedPattern::new(
            PatternKind::AgeToAge,
            vec![(12, 1.6), (24, 1.25)],
        ))
    }
    .fit_diagonal(&d)
    .unwrap();
    let expected: Vec<f64> = cl.ultimate.iter().map(|u| u * 1.05).collect();
    assert_close(&with_tail.ultimate, &expected, "age-to-age with a tail");
}

#[test]
fn quarters_and_months_take_the_grid_they_need() {
    // Accident quarters of 2024 at December 2024: 12, 9, 6 and 3 months.
    let origins = [1, 4, 7, 10].map(|m| Month::new(2024, m).unwrap());
    let d = LatestDiagonal::at_valuation(
        Grain::Quarter,
        &origins,
        Month::new(2024, 12).unwrap(),
        &[400.0, 300.0, 200.0, 100.0],
    )
    .unwrap();
    assert_eq!(d.ages(), [12, 9, 6, 3]);
    let pattern = SelectedPattern::new(
        PatternKind::AgeToUltimate,
        vec![(3, 4.0), (12, 2.0), (24, 1.25), (36, 1.0)],
    );
    let cl = selected(pattern).fit_diagonal(&d).unwrap();
    assert_eq!(cl.development.development[..3], [3, 6, 9]);
    let between = |age: f64| 4.0 * 0.5f64.powf((age - 3.0) / 9.0);
    assert_close(
        &cl.ultimate,
        &[800.0, 300.0 * between(9.0), 200.0 * between(6.0), 400.0],
        "quarters",
    );

    // Accident years at January 2025: 25 and 13 months, on a monthly grid.
    let d = LatestDiagonal::at_valuation(
        Grain::Year,
        &[2023, 2024].map(Month::january),
        Month::new(2025, 1).unwrap(),
        &[500.0, 200.0],
    )
    .unwrap();
    assert_eq!(d.ages(), [25, 13]);
    let cl = selected(annual_cdf()).fit_diagonal(&d).unwrap();
    assert_eq!(cl.development.development[..2], [12, 13]);
    assert_close(
        &cl.ultimate,
        &[
            500.0 * 1.25 * 0.8f64.powf(1.0 / 12.0),
            200.0 * 2.0 * 0.625f64.powf(1.0 / 12.0),
        ],
        "months",
    );
}

#[test]
fn the_types_ages_from_a_valuation_are_the_triangles() {
    let origins = [2022, 2023, 2024].map(Month::january);
    let valuation = Month::new(2024, 9).unwrap();
    let d =
        LatestDiagonal::at_valuation(Grain::Year, &origins, valuation, &[1.0, 2.0, 3.0]).unwrap();
    let tri = Triangle::from_long(&Long {
        keys: &[],
        origin: &origins,
        development: DevelopmentColumn::Valuation(&[valuation; 3]),
        values: &[("paid", &[1.0, 2.0, 3.0])],
        origin_grain: Grain::Year,
        development_grain: Grain::Year,
        cumulative: true,
    })
    .unwrap();
    let from_triangle: Vec<Lag> = (0..3)
        .map(|o| {
            let d = (0..tri.development().len())
                .find(|&d| tri.get(0, 0, o, d).is_some())
                .unwrap();
            tri.development()[d]
        })
        .collect();
    assert_eq!(d.ages(), from_triangle);
    assert_eq!(d.ages(), [33, 21, 9]);
}

#[test]
fn the_types_errors_name_their_cause() {
    let diag = |reason: &str, e: Error| match e {
        Error::Diagonal(why) => assert!(why.contains(reason), "{why}"),
        other => panic!("{other} is not a latest-diagonal error"),
    };
    let h = hand();
    let origins: Vec<Month> = (2019..=2024).map(Month::january).collect();
    let ages: [Lag; 6] = [72, 60, 48, 36, 24, 12];
    let d = LatestDiagonal::new(Grain::Year, &origins, &ages, &h.latest).unwrap();

    diag(
        "a selected pattern is needed",
        ChainLadder::default().fit_diagonal(&d).unwrap_err(),
    );
    let blended = hand_pattern(&h).with_weight(PatternWeight::All(0.5));
    diag(
        "fully selected",
        selected(blended).fit_diagonal(&d).unwrap_err(),
    );
    let windowed = ChainLadder {
        development: Development {
            pattern: Some(hand_pattern(&h)),
            n_periods: Some(3),
            ..Default::default()
        },
        ..Default::default()
    };
    diag("n_periods", windowed.fit_diagonal(&d).unwrap_err());
    let late = SelectedPattern::new(
        PatternKind::AgeToUltimate,
        vec![(24, 1.3), (60, 1.02), (72, 1.0)],
    );
    diag(
        "origin 2024 is at 12 months, outside the pattern's ages from 24 to 72 months",
        selected(late).fit_diagonal(&d).unwrap_err(),
    );
    let early = SelectedPattern::new(PatternKind::AgeToAge, vec![(12, 1.3), (24, 1.15)]);
    diag(
        "outside the pattern's ages from 12 to 36 months",
        selected(early).fit_diagonal(&d).unwrap_err(),
    );
    diag(
        "need an exposure",
        BornhuetterFerguson {
            apriori: 0.7.into(),
            chain_ladder: selected(hand_pattern(&h)),
        }
        .fit_diagonal(&d)
        .unwrap_err(),
    );
    let twice = [2023, 2024, 2024].map(Month::january);
    diag(
        "origin 2024 is given twice",
        LatestDiagonal::new(Grain::Year, &twice, &[24, 12, 12], &[1.0, 2.0, 3.0]).unwrap_err(),
    );
    assert_eq!(
        LatestDiagonal::new(Grain::Year, &origins[..2], &[12, 0], &[1.0, 2.0]).unwrap_err(),
        Error::NonPositiveAge { row: 1 }
    );
    assert!(matches!(
        d.clone()
            .with_exposure(&[1.0, 1.0, 0.0, 1.0, 1.0, 1.0])
            .unwrap_err(),
        Error::InvalidExposure { .. }
    ));
    // A skipped origin has no latest value.
    let gap = LatestDiagonal::new(
        Grain::Year,
        &[2022, 2024].map(Month::january),
        &[36, 12],
        &[900.0, 150.0],
    )
    .unwrap();
    assert!(matches!(
        selected(annual_cdf()).fit_diagonal(&gap).unwrap_err(),
        Error::EmptyOrigin(_)
    ));
}

#[test]
fn an_estimated_tail_on_a_finer_grid_is_the_patterns_own() {
    // GenIns' own age-to-age factors and an estimated tail, read on a
    // half-year grid: the tail is the one fitted at the annual ages.
    let full = triangle("genins");
    let [ldf, _] = own_patterns(&full, "values");
    let latest: Vec<f64> = (0..10).map(|k| 1000.0 * f64::from(k + 1)).collect();
    let annual: Vec<Month> = (2001..=2010).map(Month::january).collect();
    let ages: Vec<Lag> = (1..=10).rev().map(|k| 12 * k).collect();
    let on_years = LatestDiagonal::new(Grain::Year, &annual, &ages, &latest).unwrap();
    let mut semester_ages = ages.clone();
    semester_ages[9] = 6;
    let on_semesters = LatestDiagonal::new(Grain::Semester, &annual, &semester_ages, &latest);
    // Annual origins on a semester grain are not consecutive, so take
    // the half-year origins of a short diagonal instead.
    assert!(on_semesters.is_ok());
    let half: Vec<Month> = [(2009, 1), (2009, 7), (2010, 1)]
        .map(|(y, m)| Month::new(y, m).unwrap())
        .to_vec();
    let short = LatestDiagonal::new(
        Grain::Semester,
        &half,
        &[24, 18, 12],
        &[500.0, 300.0, 100.0],
    )
    .unwrap();
    for tail in [
        Tail::LogLinear,
        Tail::Curve(Default::default()),
        Tail::Bondy(Default::default()),
    ] {
        let cl = ChainLadder {
            tail,
            ..selected(ldf.clone())
        };
        let own = cl.fit_diagonal(&on_years).unwrap();
        let finer = cl.fit_diagonal(&short).unwrap();
        assert!(own.tail.factor > 1.0, "{tail:?}");
        assert!(
            (finer.tail.factor - own.tail.factor).abs() < 1e-12,
            "{tail:?}: {} against {}",
            finer.tail.factor,
            own.tail.factor
        );
        // The factors turned to ultimate carry no tail of their own.
        assert_eq!(finer.pattern_tail, None, "{tail:?}");
        assert_eq!(finer.pattern_rescale, None, "{tail:?}");
    }
}

#[test]
fn the_types_pattern_errors_name_the_factor_given() {
    let half = [(2023, 1), (2023, 7), (2024, 1)].map(|(y, m)| Month::new(y, m).unwrap());
    let d = LatestDiagonal::new(
        Grain::Semester,
        &half,
        &[24, 18, 12],
        &[500.0, 300.0, 100.0],
    )
    .unwrap();
    let pattern = |values: Vec<(Lag, f64)>| SelectedPattern::new(PatternKind::AgeToAge, values);
    let at = |e: Error| match e {
        Error::Pattern { age, reason } => (age, reason),
        other => panic!("{other} is not a pattern error"),
    };
    assert_eq!(
        at(selected(pattern(vec![(12, 1.6), (24, -1.0)]))
            .fit_diagonal(&d)
            .unwrap_err()),
        (Some(24), "an age-to-age factor must be finite and positive")
    );
    assert_eq!(
        at(selected(pattern(vec![(12, 1.6), (12, 1.2)]))
            .fit_diagonal(&d)
            .unwrap_err()),
        (Some(12), "the age is given twice")
    );
    let by_age = |weights: Vec<(Lag, f64)>| {
        selected(pattern(vec![(12, 1.6), (24, 1.25)]).with_weight(PatternWeight::ByAge(weights)))
    };
    assert_eq!(
        at(by_age(vec![(18, 1.0)]).fit_diagonal(&d).unwrap_err()),
        (
            Some(18),
            "a weight is given at an age with no selected factor"
        )
    );
    assert_eq!(
        at(by_age(vec![(12, 1.0), (12, 1.0)])
            .fit_diagonal(&d)
            .unwrap_err()),
        (Some(12), "the weight's age is given twice")
    );
    // Weights of 1 at the ages given are a fully selected pattern.
    let full = by_age(vec![(12, 1.0), (24, 1.0)]).fit_diagonal(&d).unwrap();
    let plain = selected(pattern(vec![(12, 1.6), (24, 1.25)]))
        .fit_diagonal(&d)
        .unwrap();
    assert_close(&full.ultimate, &plain.ultimate, "weights of 1 by age");
}

#[test]
fn the_types_other_errors_name_their_cause() {
    let diag = |reason: &str, e: Error| match e {
        Error::Diagonal(why) => assert!(why.contains(reason), "{why}"),
        other => panic!("{other} is not a latest-diagonal error"),
    };
    let h = hand();
    let origins: Vec<Month> = (2019..=2024).map(Month::january).collect();
    let ages: [Lag; 6] = [72, 60, 48, 36, 24, 12];
    let d = LatestDiagonal::new(Grain::Year, &origins, &ages, &h.latest).unwrap();
    let dropped = ChainLadder {
        development: Development {
            pattern: Some(hand_pattern(&h)),
            drop_high: 1,
            ..Default::default()
        },
        ..Default::default()
    };
    diag("the drops", dropped.fit_diagonal(&d).unwrap_err());
    assert!(matches!(
        d.clone()
            .with_exposure(&[1.0, 1.0, f64::NAN, 1.0, 1.0, 1.0])
            .unwrap_err(),
        Error::InvalidExposure { .. }
    ));
    // A monthly grid over two centuries is refused.
    let far =
        LatestDiagonal::new(Grain::Month, &[Month::new(2024, 1).unwrap()], &[1], &[1.0]).unwrap();
    let long = SelectedPattern::new(PatternKind::AgeToUltimate, vec![(1, 2.0), (3000, 1.0)]);
    diag(
        "more than 2400 development periods",
        selected(long).fit_diagonal(&far).unwrap_err(),
    );
}

#[test]
fn an_origin_at_the_end_of_the_factors_takes_the_tail_alone() {
    // Age-to-age factors from 12 to 24 and 24 to 36 months span 12 to 36:
    // an origin at 36 months takes only the tail.
    let origins = [2022, 2023, 2024].map(Month::january);
    let d =
        LatestDiagonal::new(Grain::Year, &origins, &[36, 24, 12], &[900.0, 600.0, 150.0]).unwrap();
    let cl = ChainLadder {
        tail: Tail::from(1.05),
        ..selected(SelectedPattern::new(
            PatternKind::AgeToAge,
            vec![(12, 1.6), (24, 1.25)],
        ))
    }
    .fit_diagonal(&d)
    .unwrap();
    assert_close(
        &cl.ultimate,
        &[900.0 * 1.05, 600.0 * 1.25 * 1.05, 150.0 * 2.0 * 1.05],
        "the tail alone at the factors' end",
    );
}

/// Issue #632's triangles: `paid` at each origin's own age, 1,500 of
/// premium each.
fn issue_632(origins: &[i32], ages: &[Lag], paid: &[f64]) -> Triangle {
    let origin: Vec<Month> = origins.iter().map(|&y| Month::january(y)).collect();
    let premium = vec![1500.0; paid.len()];
    Triangle::from_long(&Long {
        keys: &[],
        origin: &origin,
        development: DevelopmentColumn::Age(ages),
        values: &[("paid", paid), ("premium", &premium)],
        origin_grain: Grain::Year,
        development_grain: Grain::Year,
        cumulative: true,
    })
    .unwrap()
}

/// The issue's pattern to ultimate, from 12 to 60 months.
const ISSUE_632_CDF: [(Lag, f64); 5] = [(12, 3.0), (24, 1.8), (36, 1.35), (48, 1.12), (60, 1.0)];

/// The chain ladder, Bornhuetter-Ferguson and the expected loss method at
/// an ELR of 0.7, Benktander's two iterations and Cape Cod on `tri` with
/// `pattern` at weight 1, against the hand formulas with each origin's
/// factor to ultimate `cdf`.
fn the_five_methods_by_hand(tri: &Triangle, pattern: SelectedPattern, paid: &[f64], cdf: &[f64]) {
    let chain_ladder = selected(pattern.with_weight(PatternWeight::All(1.0)));
    let (elr, premium) = (0.7, 1500.0);
    let reported: Vec<f64> = cdf.iter().map(|c| 1.0 / c).collect();
    let cl = chain_ladder.fit(tri, "paid").unwrap();
    let expected: Vec<f64> = paid.iter().zip(cdf).map(|(l, c)| l * c).collect();
    assert_close(&cl.ultimate, &expected, "chain ladder");
    let bf_by_hand: Vec<f64> = paid
        .iter()
        .zip(&reported)
        .map(|(l, p)| l + elr * premium * (1.0 - p))
        .collect();
    let bf = BornhuetterFerguson {
        apriori: elr.into(),
        chain_ladder: chain_ladder.clone(),
    }
    .fit(tri, "paid", "premium")
    .unwrap();
    assert_close(&bf.ultimate, &bf_by_hand, "Bornhuetter-Ferguson");
    let el = ExpectedLoss {
        apriori: elr.into(),
        chain_ladder: chain_ladder.clone(),
    }
    .fit(tri, "paid", "premium")
    .unwrap();
    assert_close(
        &el.ultimate,
        &vec![elr * premium; paid.len()],
        "expected loss",
    );
    let bk = Benktander {
        apriori: elr.into(),
        n_iters: 2,
        chain_ladder: chain_ladder.clone(),
    }
    .fit(tri, "paid", "premium")
    .unwrap();
    let expected: Vec<f64> = paid
        .iter()
        .zip(&reported)
        .zip(&bf_by_hand)
        .map(|((l, p), bf)| l + bf * (1.0 - p))
        .collect();
    assert_close(&bk.ultimate, &expected, "Benktander");
    let cc = CapeCod {
        trend: 0.0,
        decay: 1.0,
        chain_ladder,
    }
    .fit(tri, "paid", "premium")
    .unwrap();
    let cc_elr = paid.iter().sum::<f64>() / reported.iter().map(|p| premium * p).sum::<f64>();
    let expected: Vec<f64> = paid
        .iter()
        .zip(&reported)
        .map(|(l, p)| l + cc_elr * premium * (1.0 - p))
        .collect();
    assert_close(&cc.expected_loss.ultimate, &expected, "Cape Cod");
}

#[test]
fn issue_632_a_diagonal_only_triangle_fits_with_a_selected_cdf() {
    // Five accident years, one cell each, at 60, 48, 36, 24 and 12 months.
    let paid = [1000.0, 800.0, 600.0, 400.0, 200.0];
    let tri = issue_632(
        &[2020, 2021, 2022, 2023, 2024],
        &[60, 48, 36, 24, 12],
        &paid,
    );
    let cdf = SelectedPattern::new(PatternKind::AgeToUltimate, ISSUE_632_CDF.to_vec());
    the_five_methods_by_hand(&tri, cdf.clone(), &paid, &[1.0, 1.12, 1.35, 1.8, 3.0]);
    let cl = selected(cdf.clone()).fit(&tri, "paid").unwrap();
    assert_close(
        &cl.ultimate,
        &[1000.0, 896.0, 810.0, 720.0, 600.0],
        "the ultimates",
    );
    let bf = BornhuetterFerguson {
        apriori: 0.7.into(),
        chain_ladder: selected(cdf),
    }
    .fit(&tri, "paid", "premium")
    .unwrap();
    assert_close(&bf.ultimate[..2], &[1000.0, 912.5], "the BF ultimates");
    assert!((bf.ultimate[4] - 900.0).abs() < 1e-9);
}

#[test]
fn issue_632_one_origin_at_one_age_fits_with_a_full_or_a_one_value_cdf() {
    let tri = issue_632(&[2024], &[12], &[200.0]);
    let full = SelectedPattern::new(PatternKind::AgeToUltimate, ISSUE_632_CDF.to_vec());
    the_five_methods_by_hand(&tri, full.clone(), &[200.0], &[3.0]);
    // One value to ultimate at the only age is the tail from it (decision
    // 20), as a factor to ultimate or a percent developed.
    let one = SelectedPattern::new(PatternKind::AgeToUltimate, vec![(12, 3.0)]);
    the_five_methods_by_hand(&tri, one.clone(), &[200.0], &[3.0]);
    let third = SelectedPattern::new(PatternKind::PercentDeveloped, vec![(12, 1.0 / 3.0)]);
    the_five_methods_by_hand(&tri, third, &[200.0], &[3.0]);
    let fit = selected(one).fit(&tri, "paid").unwrap();
    assert!((fit.ultimate[0] - 600.0).abs() < 1e-9);
    assert_eq!(fit.pattern_tail, Some(3.0));
    assert_eq!(fit.pattern_rescale, Some(1.0));
    // So does the latest diagonal type, the same fit.
    let d = LatestDiagonal::new(Grain::Year, &[Month::january(2024)], &[12], &[200.0]).unwrap();
    let one = SelectedPattern::new(PatternKind::AgeToUltimate, vec![(12, 3.0)]);
    assert_eq!(
        selected(one.clone()).fit_diagonal(&d).unwrap().ultimate,
        fit.ultimate
    );
    // No development past the value's age is a tail of 1 there, which
    // would leave the value inert.
    let none = d.with_past_pattern_end(PastPatternEnd::NoDevelopment);
    match selected(one).fit_diagonal(&none) {
        Err(Error::Pattern { reason, .. }) => {
            assert!(reason.contains("a tail is given"), "{reason}")
        }
        other => panic!("{other:?} is not a pattern error"),
    }
    // One origin at 48 months takes the full pattern from 48 months on.
    let older = issue_632(&[2021], &[48], &[800.0]);
    the_five_methods_by_hand(&older, full, &[800.0], &[1.12]);
}

#[test]
fn issue_632_a_pattern_that_ends_at_the_only_age_is_its_level() {
    // A pattern that ends at an origin's only age selects no link and is
    // its level there, the tail, as one value at that age is (decision 20
    // item 1).
    let full = SelectedPattern::new(PatternKind::AgeToUltimate, ISSUE_632_CDF.to_vec());
    let at_60 = issue_632(&[2020], &[60], &[1000.0]);
    the_five_methods_by_hand(&at_60, full, &[1000.0], &[1.0]);
    let to_48 = SelectedPattern::new(PatternKind::AgeToUltimate, ISSUE_632_CDF[..4].to_vec());
    let at_48 = issue_632(&[2021], &[48], &[800.0]);
    the_five_methods_by_hand(&at_48, to_48.clone(), &[800.0], &[1.12]);
    let fit = selected(to_48).fit(&at_48, "paid").unwrap();
    assert!((fit.ultimate[0] - 896.0).abs() < 1e-9);
    assert_eq!(fit.pattern_tail, Some(1.12));
    let from_6 = SelectedPattern::new(PatternKind::AgeToUltimate, vec![(6, 4.0), (12, 3.0)]);
    let at_12 = issue_632(&[2024], &[12], &[200.0]);
    the_five_methods_by_hand(&at_12, from_6, &[200.0], &[3.0]);
}

#[test]
fn issue_632_one_value_with_a_tail_past_its_end_is_an_error() {
    // Decision 20 item 3: an origin older than the value's age, with a
    // tail or no development past the end, would rescale the one value
    // away, so it is the same error as with every origin at its age.
    let d = LatestDiagonal::new(
        Grain::Year,
        &[Month::january(2023), Month::january(2024)],
        &[24, 12],
        &[400.0, 200.0],
    )
    .unwrap();
    for kind in [PatternKind::AgeToUltimate, PatternKind::PercentDeveloped] {
        let value = if kind == PatternKind::AgeToUltimate {
            3.0
        } else {
            1.0 / 3.0
        };
        let one = SelectedPattern::new(kind, vec![(12, value)]);
        for (rule, tail) in [
            (PastPatternEnd::Tail, Tail::from(1.5)),
            (PastPatternEnd::NoDevelopment, Tail::default()),
        ] {
            let cl = ChainLadder {
                tail,
                ..selected(one.clone())
            };
            match cl.fit_diagonal(&d.clone().with_past_pattern_end(rule)) {
                Err(Error::Pattern { age, reason }) => {
                    assert_eq!(age, Some(12));
                    assert!(reason.contains("a tail is given"), "{reason}");
                }
                other => panic!("{rule:?}: {other:?} is not a pattern error"),
            }
        }
    }
}
