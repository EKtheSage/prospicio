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

use prospicio_reserving::{
    Benktander, BornhuetterFerguson, CapeCod, ChainLadder, Development, DevelopmentColumn, Error,
    ExpectedLoss, Grain, Lag, Long, Mack, MackBootstrap, Month, OdpBootstrap, PatternKind,
    SelectedPattern, Tail, Triangle,
};
use prospicio_validation::{triangle, triangle_columns};

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
        apriori: elr,
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
        apriori: elr,
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
        apriori: 0.7,
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
                    apriori: 0.6,
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
                    apriori: 0.6,
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
fn a_pattern_age_before_the_youngest_origin_is_off_the_grid() {
    // The youngest origin is at 24 months, so the grid starts there and a
    // pattern from 12 months does not fit it.
    let h = hand();
    let tri = rows(&[
        (2019, 72, h.latest[0]),
        (2020, 60, h.latest[1]),
        (2021, 48, h.latest[2]),
        (2022, 36, h.latest[3]),
        (2023, 24, h.latest[4]),
    ]);
    let e = selected(hand_pattern(&h)).fit(&tri, "paid").unwrap_err();
    assert!(matches!(e, Error::Pattern { age: Some(12), .. }), "{e}");
}
