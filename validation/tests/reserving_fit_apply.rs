//! A development fitted on one book and applied to another or to a latest
//! diagonal (decision 17 of `docs/design/reserving-v02.md`, issue #544):
//! `ChainLadderFit::to_chain_ladder` and `LatestDiagonal::from_triangle`.
//!
//! * A. Round trip: GenIns and RAA fitted with each average, `n_periods`,
//!   the drops, a blended, a cumulative and a percent-developed pattern and
//!   each tail (an early attachment too), then applied to their own latest
//!   diagonal and to their own triangle, give the same factors to the bit
//!   and the same factors to ultimate and ultimates within 1e-14 (to the bit
//!   with no tail), and so do the segments of a `ChainLadder::fit_segments`;
//!   the expected-loss methods on the CAS workers' compensation triangle
//!   give the same development and exposure to the bit and the same
//!   ultimates within 1e-12 (Cape Cod's aprioris too, its premium running no
//!   later than its paid), and a fit made on a diagonal applies again to it.
//! * B. chainladder-python's `fit(source).predict(target)`
//!   (`reference/reserving_fit_apply_python.csv`, from
//!   `scripts/reserving_fit_apply_python.py`) in two arms: (i) our own source
//!   fit through `to_chain_ladder`, and (ii) the reference's own `cdf_` as a
//!   pattern of factors to ultimate, which checks the apply path apart from
//!   the source fit. GenIns with the chain ladder and each tail applied to
//!   its diagonal and a perturbed book, whose own curve tails (refitted in
//!   arm (i)) differ from the carried ones by more than the tolerance; the
//!   source with no tail also applied to a doubled and a shifted book and
//!   two subsets of its origins, and to a book valued before 2008, whose
//!   reference is `DevelopmentConstant` with the source's `cdf_`, not
//!   predict. The CAS workers' compensation triangle with
//!   Bornhuetter-Ferguson, Benktander and Cape Cod (its apriori estimated
//!   again on the target) applied to its diagonal, a perturbed and a shifted
//!   book; Cape Cod also to the origins from 1993 and to the raw latest
//!   diagonal, whose apriori predict reads with every origin at the first
//!   age, reproduced so. The GenIns targets at the source's ages are also
//!   fitted by the plain triangle path, against the same rows.
//! * C. By hand, where chainladder-python has no reference: ages between the
//!   source's (18 months, accident quarters) by the log-linear rule, a target
//!   with fewer ages through the diagonal and its error through the plain
//!   triangle path, a target with more ages through the plain path (its own
//!   factors past the source's oldest age, pinned), an origin with its
//!   latest diagonal cell missing (where chainladder-python gives NaN), the
//!   applied fit's selection, and Cape Cod on a diagonal trended to the
//!   fitted column's own valuation when the premium runs later.
//! * D. Errors: ages outside the source's, the errors of `from_triangle`, the
//!   expected-loss methods without an exposure, a carried tail below the
//!   floor of exp(-0.75) (one just above it is spread) and a source with one
//!   age.
//! * E. The doctests of `ChainLadderFit::to_chain_ladder` and
//!   `LatestDiagonal::from_triangle`, with exact numbers, run by
//!   `cargo test -p prospicio-reserving --doc`.

use std::collections::HashMap;

use prospicio_reserving::{
    Average, Benktander, BornhuetterFerguson, CapeCod, ChainLadder, ChainLadderFit, CurveShape,
    Development, DevelopmentColumn, Error, Grain, Lag, LatestDiagonal, Long, Month, PatternKind,
    PatternWeight, Period, SelectedPattern, Tail, TailBondy, TailConstant, TailCurve, Triangle,
};
use prospicio_validation::{Case, check, reference, triangle, triangle_columns};

/// A copy of the single-segment `tri` with the cells `keep` keeps (by origin
/// year and age), each value mapped by `value` (by column position, origin
/// year and value) and the origins `shift` years later.
fn remake(
    tri: &Triangle,
    shift: i32,
    keep: impl Fn(i32, Lag) -> bool,
    value: impl Fn(usize, i32, f64) -> f64,
) -> Triangle {
    let ages = tri.development();
    let columns = tri.columns().to_vec();
    let mut origin = Vec::new();
    let mut age = Vec::new();
    let mut values = vec![Vec::new(); columns.len()];
    for (o, p) in tri.origins().iter().enumerate() {
        let year = p.start().year();
        for (d, &a) in ages.iter().enumerate() {
            let cells: Option<Vec<f64>> = (0..columns.len()).map(|c| tri.get(0, c, o, d)).collect();
            let Some(cells) = cells.filter(|_| keep(year, a)) else {
                continue;
            };
            origin.push(Month::january(year + shift));
            age.push(a);
            for (c, v) in cells.into_iter().enumerate() {
                values[c].push(value(c, year, v));
            }
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

fn close_to(found: f64, expected: f64, rel: f64) -> bool {
    found == expected || (found - expected).abs() <= rel * expected.abs()
}

fn assert_close(found: &[f64], expected: &[f64], rel: f64, what: &str) {
    assert_eq!(found.len(), expected.len(), "{what}: lengths");
    for (k, (f, e)) in found.iter().zip(expected).enumerate() {
        assert!(
            close_to(*f, *e, rel),
            "{what}, element {k}: {f} against {e}"
        );
    }
}

// A. Round trip.

/// The source settings of the round trip on a triangle with factors
/// `ldf` and factors to ultimate `cdf` at `ages`, by name.
fn round_trip_sources(ages: &[Lag], ldf: &[f64], cdf: &[f64]) -> Vec<(&'static str, ChainLadder)> {
    let with = |development: Development| ChainLadder {
        development,
        ..Default::default()
    };
    let average = |average| {
        with(Development {
            average,
            ..Default::default()
        })
    };
    let tail = |tail: Tail| ChainLadder {
        tail,
        ..Default::default()
    };
    let blended = SelectedPattern::new(
        PatternKind::AgeToAge,
        ages.iter().zip(ldf).map(|(&a, &f)| (a, f * 1.02)).collect(),
    )
    .with_weight(PatternWeight::All(0.5));
    let to_ultimate = SelectedPattern::new(
        PatternKind::AgeToUltimate,
        ages.iter().zip(cdf).map(|(&a, &c)| (a, c * 1.03)).collect(),
    );
    let percent = SelectedPattern::new(
        PatternKind::PercentDeveloped,
        ages.iter().zip(cdf).map(|(&a, &c)| (a, 1.0 / c)).collect(),
    );
    vec![
        ("volume", average(Average::Volume)),
        ("simple", average(Average::Simple)),
        ("regression", average(Average::Regression)),
        ("geometric", average(Average::Geometric)),
        (
            "n_periods = 5",
            with(Development {
                n_periods: Some(5),
                ..Default::default()
            }),
        ),
        (
            "drop_high = drop_low = 1",
            with(Development {
                drop_high: 1,
                drop_low: 1,
                ..Default::default()
            }),
        ),
        ("blended age-to-age at 0.5", selected(blended)),
        ("to ultimate with its own tail", selected(to_ultimate)),
        (
            "percent developed rescaled to a tail",
            ChainLadder {
                tail: Tail::from(1.05),
                ..selected(percent)
            },
        ),
        ("TailConstant(1.05)", tail(Tail::from(1.05))),
        ("TailCurve exponential", tail(TailCurve::default().into())),
        (
            "TailCurve inverse power",
            tail(
                TailCurve {
                    curve: CurveShape::InversePower,
                    ..Default::default()
                }
                .into(),
            ),
        ),
        ("Bondy", tail(TailBondy::default().into())),
        (
            "TailCurve attached at 84 months",
            tail(
                TailCurve {
                    attachment_age: Some(84),
                    ..Default::default()
                }
                .into(),
            ),
        ),
    ]
}

/// `applied` projects as `source`: the factors to the bit, the factors to
/// ultimate and the ultimates within 1e-14 relative, and to the bit when the
/// tail factor is 1.
fn assert_same_projection(applied: &ChainLadderFit, source: &ChainLadderFit, what: &str) {
    assert_eq!(applied.origins, source.origins, "{what}: origins");
    assert_eq!(applied.ldf(), source.ldf(), "{what}: factors");
    if source.tail.factor == 1.0 {
        assert_eq!(applied.tail.factor, 1.0, "{what}: tail");
        assert_eq!(applied.cdf, source.cdf, "{what}: factors to ultimate");
        assert_eq!(applied.ultimate, source.ultimate, "{what}: ultimates");
    } else {
        // The carried tail is spread again, so its product, the tail
        // factor, can move by an ulp or so.
        assert!(
            close_to(applied.tail.factor, source.tail.factor, 1e-14),
            "{what}: tail {} against {}",
            applied.tail.factor,
            source.tail.factor
        );
        assert_close(&applied.cdf, &source.cdf, 1e-14, what);
        assert_close(&applied.ultimate, &source.ultimate, 1e-14, what);
    }
}

#[test]
fn a_fit_applied_to_its_own_book_projects_as_it_did() {
    for name in ["genins", "raa"] {
        let tri = triangle(name);
        let own = ChainLadder::default().fit(&tri, "values").unwrap();
        let ages = tri.development().to_vec();
        let diagonal = LatestDiagonal::from_triangle(&tri, "values", None).unwrap();
        for (label, method) in round_trip_sources(&ages, own.ldf(), &own.cdf) {
            let what = format!("{name}, {label}");
            let source = method.fit(&tri, "values").unwrap();
            match label {
                "to ultimate with its own tail" => {
                    assert!(source.pattern_tail.is_some(), "{what}")
                }
                "percent developed rescaled to a tail" => {
                    assert!(source.pattern_rescale.is_some(), "{what}")
                }
                _ => {}
            }
            let frozen = source.to_chain_ladder();
            assert_same_projection(
                &frozen.fit_diagonal(&diagonal).unwrap(),
                &source,
                &format!("{what}, on the diagonal"),
            );
            assert_same_projection(
                &frozen.fit(&tri, "values").unwrap(),
                &source,
                &format!("{what}, on the triangle"),
            );
        }
    }
}

#[test]
fn the_expected_loss_methods_carry_the_development() {
    let tri = triangle_columns("clrd_wkcomp", &["paid", "premium"]);
    let diagonal = LatestDiagonal::from_triangle(&tri, "paid", Some("premium")).unwrap();
    let rel = 1e-12;
    let cl = ChainLadder::default();

    let bf = BornhuetterFerguson {
        apriori: 0.6.into(),
        chain_ladder: cl.clone(),
    };
    let source = bf.fit(&tri, "paid", "premium").unwrap();
    let frozen = BornhuetterFerguson {
        chain_ladder: source.chain_ladder.to_chain_ladder(),
        ..bf
    };
    for applied in [
        frozen.fit_diagonal(&diagonal).unwrap(),
        frozen.fit(&tri, "paid", "premium").unwrap(),
    ] {
        assert_same_projection(&applied.chain_ladder, &source.chain_ladder, "BF");
        assert_eq!(applied.exposure, source.exposure, "BF exposure");
        assert_close(&applied.ultimate, &source.ultimate, rel, "BF ultimates");
    }

    let bk = Benktander {
        apriori: 0.6.into(),
        n_iters: 3,
        chain_ladder: cl.clone(),
    };
    let source = bk.fit(&tri, "paid", "premium").unwrap();
    let frozen = Benktander {
        chain_ladder: source.chain_ladder.to_chain_ladder(),
        ..bk
    };
    for applied in [
        frozen.fit_diagonal(&diagonal).unwrap(),
        frozen.fit(&tri, "paid", "premium").unwrap(),
    ] {
        assert_same_projection(&applied.chain_ladder, &source.chain_ladder, "Benktander");
        assert_eq!(applied.exposure, source.exposure, "Benktander exposure");
        assert_close(&applied.ultimate, &source.ultimate, rel, "Benktander");
    }

    for (trend, decay) in [(0.0, 1.0), (0.02, 0.75)] {
        let cc = CapeCod {
            trend,
            decay,
            chain_ladder: cl.clone(),
        };
        let source = cc.fit(&tri, "paid", "premium").unwrap();
        let frozen = CapeCod {
            chain_ladder: source.expected_loss.chain_ladder.to_chain_ladder(),
            ..cc
        };
        for applied in [
            frozen.fit_diagonal(&diagonal).unwrap(),
            frozen.fit(&tri, "paid", "premium").unwrap(),
        ] {
            let what = format!("Cape Cod at trend {trend} and decay {decay}");
            let (a, s) = (&applied.expected_loss, &source.expected_loss);
            assert_same_projection(&a.chain_ladder, &s.chain_ladder, &what);
            assert_eq!(a.exposure, s.exposure, "{what}: exposure");
            assert_close(&a.ultimate, &s.ultimate, rel, &what);
            assert_close(&a.apriori, &s.apriori, rel, &what);
            assert_close(
                &applied.trended_apriori,
                &source.trended_apriori,
                rel,
                &what,
            );
        }
    }
}

#[test]
fn a_segment_of_fit_segments_applies_as_it_fitted() {
    // GenIns and the perturbed GenIns as two lines of one keyed triangle.
    let books = [
        ("a", triangle("genins")),
        ("b", target("genins", "perturbed")),
    ];
    let (mut lob, mut origin, mut age, mut values) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for (label, tri) in &books {
        for (o, p) in tri.origins().iter().enumerate() {
            for (d, &a) in tri.development().iter().enumerate() {
                if let Some(v) = tri.get(0, 0, o, d) {
                    lob.push(*label);
                    origin.push(p.start());
                    age.push(a);
                    values.push(v);
                }
            }
        }
    }
    let keyed = Triangle::from_long(&Long {
        keys: &[("lob", lob.as_slice())],
        origin: &origin,
        development: DevelopmentColumn::Age(&age),
        values: &[("values", values.as_slice())],
        origin_grain: Grain::Year,
        development_grain: Grain::Year,
        cumulative: true,
    })
    .unwrap();
    let fits = ChainLadder {
        tail: TailCurve::default().into(),
        ..Default::default()
    }
    .fit_segments(&keyed, "values")
    .unwrap();
    assert_ne!(fits.fits[0].ldf(), fits.fits[1].ldf());
    for (label, book) in &books {
        let source = &fits.fits[fits.position(&[("lob", label)]).unwrap()];
        let frozen = source.to_chain_ladder();
        let d = LatestDiagonal::from_triangle(book, "values", None).unwrap();
        assert_same_projection(
            &frozen.fit_diagonal(&d).unwrap(),
            source,
            &format!("segment {label}, on the diagonal"),
        );
        assert_same_projection(
            &frozen.fit(book, "values").unwrap(),
            source,
            &format!("segment {label}, on the triangle"),
        );
    }
}

#[test]
fn a_fit_made_on_a_diagonal_applies_again_to_it() {
    let tri = triangle("genins");
    let own = ChainLadder::default().fit(&tri, "values").unwrap();
    let ages = tri.development();
    let cdf = SelectedPattern::new(
        PatternKind::AgeToUltimate,
        ages.iter().copied().zip(own.cdf.iter().copied()).collect(),
    );
    let diagonal = LatestDiagonal::from_triangle(&tri, "values", None).unwrap();
    let source = selected(cdf.clone()).fit_diagonal(&diagonal).unwrap();
    let applied = source.to_chain_ladder().fit_diagonal(&diagonal).unwrap();
    assert_close(&applied.ultimate, &source.ultimate, 1e-14, "annual");

    // On a half-year grid, from a diagonal at 24, 18 and 12 months.
    let half = [(2009, 1), (2009, 7), (2010, 1)].map(|(y, m)| Month::new(y, m).unwrap());
    let short = LatestDiagonal::new(
        Grain::Semester,
        &half,
        &[24, 18, 12],
        &[500.0, 300.0, 100.0],
    )
    .unwrap();
    let source = selected(cdf).fit_diagonal(&short).unwrap();
    assert_eq!(source.development.development[..3], [12, 18, 24]);
    let applied = source.to_chain_ladder().fit_diagonal(&short).unwrap();
    assert_close(&applied.ultimate, &source.ultimate, 1e-14, "half-year");
}

// B. chainladder-python parity.

/// A GenIns or workers' compensation target, by the name the reference
/// gives it.
fn target(dataset: &str, name: &str) -> Triangle {
    let all = |_: i32, _: Lag| true;
    let same = |_: usize, _: i32, v: f64| v;
    if dataset == "genins" {
        let tri = triangle(dataset);
        return match name {
            "diagonal" => tri,
            "perturbed" => remake(&tri, 0, all, |_, year, v| {
                v * (1.0 + 0.01 * f64::from(year - 2000))
            }),
            "double" => remake(&tri, 0, all, |_, _, v| v * 2.0),
            "from_2006" => remake(&tri, 0, |year, _| year >= 2006, same),
            "to_2004" => remake(&tri, 0, |year, _| year <= 2004, same),
            "shifted" => remake(&tri, 10, all, same),
            "early" => remake(&tri, 0, |year, age| year + age as i32 / 12 - 1 < 2008, same),
            other => panic!("unknown target {other}"),
        };
    }
    let tri = triangle_columns(dataset, &["paid", "premium"]);
    match name {
        "diagonal" | "diagonal_raw" => tri,
        // Paid and premium times 1 + 0.01 (origin - 1987), and the premium
        // times 1.1 more.
        "perturbed" => remake(&tri, 0, all, |c, year, v| {
            let v = v * (1.0 + 0.01 * f64::from(year - 1987));
            if c == 1 { v * 1.1 } else { v }
        }),
        "shifted" => remake(&tri, 10, all, same),
        "from_1993" => remake(&tri, 0, |year, _| year >= 1993, same),
        other => panic!("unknown target {other}"),
    }
}

/// The GenIns source's tail, by the reference's method.
fn genins_tail(method: &Case) -> Tail {
    let m = method.get("method");
    if m.contains("tail=none") {
        Tail::default()
    } else if m.contains("tail=constant") {
        Tail::from(method.param("method", "factor"))
    } else if m.contains("curve=inverse_power") {
        TailCurve {
            curve: CurveShape::InversePower,
            ..Default::default()
        }
        .into()
    } else {
        TailCurve::default().into()
    }
}

/// What a case reads: per origin, a total, a tail or a pattern value.
struct Applied {
    origins: Vec<Period>,
    ultimate: Vec<f64>,
    apriori: Option<Vec<f64>>,
}

/// The source's development as the arm carries it: (i) our own fit,
/// (ii) the reference's `cdf_` with no tail.
#[derive(Clone, Copy, PartialEq)]
enum Arm {
    Own,
    Reference,
}

/// How the frozen chain ladder reaches a target: through its latest
/// diagonal (`LatestDiagonal::from_triangle` and `fit_diagonal`), or by the
/// plain triangle path (`fit`), which is checked on the GenIns targets at
/// the source's ages only.
#[derive(Clone, Copy, PartialEq)]
enum Route {
    Diagonal,
    Triangle,
}

fn parity(arm: Arm, route: Route) {
    let cases = reference("reserving_fit_apply_python.csv");
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
    // The second arm takes the patterns as given, so it checks the applied
    // fits alone.
    let cases: Vec<Case> = cases
        .into_iter()
        .filter(|c| arm == Arm::Own || c.get("method").contains(";target="))
        .collect();
    // The plain triangle path reads the GenIns targets whose development
    // ages are the source's.
    let cases: Vec<Case> = match route {
        Route::Diagonal => cases,
        Route::Triangle => {
            let ages = triangle("genins").development().to_vec();
            let cases: Vec<Case> = cases
                .into_iter()
                .filter(|c| {
                    c.get("dataset") == "genins"
                        && matches!(c.get("quantity"), "ultimate" | "total_ultimate")
                        && c.get("method")
                            .split_once(";target=")
                            .is_some_and(|(_, t)| target("genins", t).development() == ages)
                })
                .collect();
            let mut names: Vec<&str> = cases
                .iter()
                .filter_map(|c| c.get("method").split_once(";target=").map(|(_, t)| t))
                .collect();
            names.sort_unstable();
            names.dedup();
            assert_eq!(
                names,
                ["diagonal", "double", "perturbed", "shifted", "to_2004"]
            );
            cases
        }
    };
    let mut sources: HashMap<(String, String), (ChainLadderFit, ChainLadder)> = HashMap::new();
    let mut applied: HashMap<(String, String), Applied> = HashMap::new();
    check(&cases, |case: &Case| {
        let (dataset, method) = (case.get("dataset"), case.get("method"));
        let (base, target_name) = match method.split_once(";target=") {
            Some((b, t)) => (b, Some(t)),
            None => (method, None),
        };
        let key = (dataset.to_string(), base.to_string());
        let (source, frozen) = sources.entry(key.clone()).or_insert_with(|| {
            let fit = if dataset == "genins" {
                ChainLadder {
                    tail: genins_tail(case),
                    ..Default::default()
                }
                .fit(&triangle(dataset), "values")
            } else {
                let tri = triangle_columns(dataset, &["paid", "premium"]);
                ChainLadder::default().fit(&tri, "paid")
            }
            .unwrap();
            let frozen = match arm {
                Arm::Own => fit.to_chain_ladder(),
                Arm::Reference => selected(SelectedPattern::new(
                    PatternKind::AgeToUltimate,
                    patterns[&key].clone(),
                )),
            };
            (fit, frozen)
        });
        let Some(target_name) = target_name else {
            // The source's own fit: its factors to ultimate and its tail.
            return match case.get("quantity") {
                "pattern" => {
                    let age = case.number("arg")? as Lag;
                    let d = source
                        .development
                        .development
                        .iter()
                        .position(|&a| a == age)?;
                    Some(source.cdf[d])
                }
                "tail" => Some(source.tail.factor),
                // The source's tail refitted on the perturbed book, which
                // predict does not use: it must differ from the carried tail
                // by more than the tolerance, so the perturbed target tells
                // the two apart.
                "refitted_tail_perturbed" => {
                    let refit = ChainLadder {
                        tail: genins_tail(case),
                        ..Default::default()
                    }
                    .fit(&target(dataset, "perturbed"), "values")
                    .unwrap();
                    let (carried, refitted) = (source.tail.factor, refit.tail.factor);
                    let rel = case.number("rel_tol")?;
                    assert!(
                        (refitted - carried).abs() > rel * carried,
                        "{method}: the refitted tail {refitted} is the carried {carried}"
                    );
                    Some(refitted)
                }
                _ => None,
            };
        };
        let frozen = frozen.clone();
        let fit = applied
            .entry((dataset.to_string(), method.to_string()))
            .or_insert_with(|| {
                let book = target(dataset, target_name);
                if dataset == "genins" {
                    let fit = match route {
                        Route::Diagonal => frozen.fit_diagonal(
                            &LatestDiagonal::from_triangle(&book, "values", None).unwrap(),
                        ),
                        Route::Triangle => frozen.fit(&book, "values"),
                    }
                    .unwrap();
                    return Applied {
                        origins: fit.origins,
                        ultimate: fit.ultimate,
                        apriori: None,
                    };
                }
                let mut d = LatestDiagonal::from_triangle(&book, "paid", Some("premium")).unwrap();
                if target_name == "diagonal_raw" {
                    // chainladder-python's Cape Cod on a latest diagonal not
                    // given with `val_to_dev()` reads every origin at the
                    // first age.
                    let first = vec![source.development.development[0]; d.origins().len()];
                    d = LatestDiagonal::new(d.origin_grain(), d.origins(), &first, d.latest())
                        .unwrap()
                        .with_exposure(d.exposure().unwrap())
                        .unwrap();
                }
                let param = |name| case.param("method", name);
                let (fit, apriori) = match base.split(';').next().unwrap() {
                    "bornhuetter_ferguson" => (
                        BornhuetterFerguson {
                            apriori: param("apriori").into(),
                            chain_ladder: frozen,
                        }
                        .fit_diagonal(&d)
                        .unwrap(),
                        None,
                    ),
                    "benktander" => (
                        Benktander {
                            apriori: param("apriori").into(),
                            n_iters: param("n_iters") as usize,
                            chain_ladder: frozen,
                        }
                        .fit_diagonal(&d)
                        .unwrap(),
                        None,
                    ),
                    "cape_cod" => {
                        let cc = CapeCod {
                            trend: param("trend"),
                            decay: param("decay"),
                            chain_ladder: frozen,
                        }
                        .fit_diagonal(&d)
                        .unwrap();
                        (cc.expected_loss, Some(cc.trended_apriori))
                    }
                    other => panic!("unknown method {other}"),
                };
                Applied {
                    origins: fit.chain_ladder.origins,
                    ultimate: fit.ultimate,
                    apriori,
                }
            });
        let origin = || {
            let year = case.number("arg")? as i32;
            fit.origins.iter().position(|p| *p == Period::year(year))
        };
        match case.get("quantity") {
            "total_ultimate" => Some(fit.ultimate.iter().sum()),
            "ultimate" => Some(fit.ultimate[origin()?]),
            "apriori" => Some(fit.apriori.as_ref()?[origin()?]),
            _ => None,
        }
    });
}

/// Arm (i): our own source fit, carried by `to_chain_ladder`. The source's
/// factors to ultimate and tails, and its curve tails refitted on the
/// perturbed book, are checked too; the curve tails rest on the tail parity
/// of `reserving_tails.rs`.
#[test]
fn a_fit_applied_to_another_book_matches_chainladder_python() {
    parity(Arm::Own, Route::Diagonal);
}

/// Arm (ii): the reference's own factors to ultimate as the pattern, so the
/// applied fits are checked apart from the source fit.
#[test]
fn the_reference_pattern_applied_matches_chainladder_python() {
    parity(Arm::Reference, Route::Diagonal);
}

/// The plain triangle path (`fit`) in both arms, on the GenIns targets
/// whose development ages are the source's.
#[test]
fn the_plain_triangle_path_matches_chainladder_python_at_the_sources_ages() {
    parity(Arm::Own, Route::Triangle);
    parity(Arm::Reference, Route::Triangle);
}

// C. By hand.

fn genins_source() -> ChainLadderFit {
    ChainLadder::default()
        .fit(&triangle("genins"), "values")
        .unwrap()
}

#[test]
fn ages_between_the_sources_are_read_log_linearly() {
    let source = genins_source();
    let frozen = source.to_chain_ladder();
    let cdf = |age: Lag| source.cdf[(age / 12 - 1) as usize];

    // Accident years at 30 and 18 months, on a half-year grid.
    let d = LatestDiagonal::new(
        Grain::Year,
        &[2009, 2010].map(Month::january),
        &[30, 18],
        &[2000.0, 1000.0],
    )
    .unwrap();
    let fit = frozen.fit_diagonal(&d).unwrap();
    assert_eq!(fit.development.development[..4], [12, 18, 24, 30]);
    assert_close(
        &fit.ultimate,
        &[
            2000.0 * (cdf(24) * cdf(36)).sqrt(),
            1000.0 * (cdf(12) * cdf(24)).sqrt(),
        ],
        1e-12,
        "half-year",
    );
    assert_eq!(fit.pattern_tail, None);
    assert_eq!(fit.pattern_rescale, None);

    // Accident quarters of 2009 and the first of 2010 at December 2010: 24,
    // 21, 18, 15 and 12 months.
    let origins = [(2009, 1), (2009, 4), (2009, 7), (2009, 10), (2010, 1)]
        .map(|(y, m)| Month::new(y, m).unwrap());
    let latest = [500.0, 400.0, 300.0, 200.0, 100.0];
    let d = LatestDiagonal::at_valuation(
        Grain::Quarter,
        &origins,
        Month::new(2010, 12).unwrap(),
        &latest,
    )
    .unwrap();
    assert_eq!(d.ages(), [24, 21, 18, 15, 12]);
    let fit = frozen.fit_diagonal(&d).unwrap();
    assert_eq!(fit.development.development[..5], [12, 15, 18, 21, 24]);
    let between = |share: f64| cdf(12).powf(1.0 - share) * cdf(24).powf(share);
    assert_close(
        &fit.ultimate,
        &[
            500.0 * cdf(24),
            400.0 * between(0.75),
            300.0 * between(0.5),
            200.0 * between(0.25),
            100.0 * cdf(12),
        ],
        1e-12,
        "quarters",
    );
}

#[test]
fn a_book_with_fewer_ages_is_applied_through_its_diagonal() {
    // GenIns from 2006 is at most 60 months old.
    let source = genins_source();
    let book = target("genins", "from_2006");
    assert_eq!(*book.development().last().unwrap(), 60);
    let frozen = source.to_chain_ladder();
    let d = LatestDiagonal::from_triangle(&book, "values", None).unwrap();
    let fit = frozen.fit_diagonal(&d).unwrap();
    assert_eq!(fit.origins, source.origins[5..]);
    assert_eq!(fit.ultimate, source.ultimate[5..]);
    // The plain triangle path reads the factors past the book's oldest age
    // as one constant tail there (decision 18), so it projects as the
    // diagonal does.
    let plain = frozen.fit(&book, "values").unwrap();
    assert_eq!(plain.origins, source.origins[5..]);
    assert_close(&plain.ultimate, &source.ultimate[5..], 1e-14, "plain path");
    let folded: f64 = source.ldf()[4..].iter().product();
    assert!(close_to(plain.pattern_tail.unwrap(), folded, 1e-14));
    // And with a tail carried from the source, which follows the pattern's
    // end.
    let tailed = ChainLadder {
        tail: Tail::from(1.05),
        ..Default::default()
    }
    .fit(&triangle("genins"), "values")
    .unwrap();
    let frozen = tailed.to_chain_ladder();
    let diagonal = frozen.fit_diagonal(&d).unwrap();
    assert_close(
        &diagonal.ultimate,
        &tailed.ultimate[5..],
        1e-14,
        "diagonal, tail",
    );
    let plain = frozen.fit(&book, "values").unwrap();
    assert_close(
        &plain.ultimate,
        &tailed.ultimate[5..],
        1e-14,
        "plain path, tail",
    );
    assert!(close_to(plain.pattern_tail.unwrap(), folded * 1.05, 1e-14));
}

#[test]
fn the_plain_path_on_an_older_book_mixes_in_its_own_factors() {
    // A source at most 60 months old with a tail of 1.05, applied to the
    // full GenIns by the plain triangle path: no error, but the factors from
    // 60 months on are GenIns' own and the tail attaches at 120 months, which
    // neither predict nor `fit_diagonal` (an error here, see
    // `an_age_outside_the_sources_is_an_error`) gives. This pins today's
    // behaviour; a guard (decision 17, item 11) would change it on purpose.
    let young = ChainLadder {
        tail: Tail::from(1.05),
        ..Default::default()
    }
    .fit(&target("genins", "from_2006"), "values")
    .unwrap();
    assert_eq!(young.ldf().len(), 4);
    let full = triangle("genins");
    let fit = young.to_chain_ladder().fit(&full, "values").unwrap();
    let own = genins_source();
    assert_eq!(fit.development.development, full.development());
    assert_eq!(fit.ldf()[..4], *young.ldf());
    assert_eq!(fit.ldf()[4..], own.ldf()[4..]);
    assert!(fit.development.selected[..4].iter().all(Option::is_some));
    assert!(fit.development.selected[4..].iter().all(Option::is_none));
    assert_eq!(fit.tail.attachment, 9);
    assert!(
        close_to(fit.tail.factor, 1.05, 1e-14),
        "{}",
        fit.tail.factor
    );
}

#[test]
fn an_origin_without_its_latest_cell_is_projected_from_its_last() {
    // 2005 at 72 months removed: it is projected from 60 months, where
    // chainladder-python gives NaN.
    let source = genins_source();
    let book = remake(
        &triangle("genins"),
        0,
        |year, age| !(year == 2005 && age == 72),
        |_, _, v| v,
    );
    let d = LatestDiagonal::from_triangle(&book, "values", None).unwrap();
    assert_eq!(d.ages()[4], 60);
    let fit = source.to_chain_ladder().fit_diagonal(&d).unwrap();
    let at60 = book.get(0, 0, 4, 4).unwrap();
    assert!((source.cdf[4] - 1.384_498_968_950_357_7).abs() < 1e-12);
    assert_eq!(fit.ultimate[4], at60 * source.cdf[4]);
    for o in (0..10).filter(|&o| o != 4) {
        assert_eq!(fit.ultimate[o], source.ultimate[o], "origin {o}");
    }
}

#[test]
fn a_book_valued_earlier_is_projected_at_its_own_ages() {
    // chainladder-python's predict values the book at the source's
    // valuation and gives NaN; the reference rows (target `early`) come from
    // `DevelopmentConstant` instead.
    let source = genins_source();
    let book = target("genins", "early");
    let d = LatestDiagonal::from_triangle(&book, "values", None).unwrap();
    assert_eq!(d.ages(), [84, 72, 60, 48, 36, 24, 12]);
    let fit = source.to_chain_ladder().fit_diagonal(&d).unwrap();
    for (o, (&u, &age)) in fit.ultimate.iter().zip(d.ages()).enumerate() {
        let latest = d.latest()[o];
        assert_eq!(
            u,
            latest * source.cdf[(age / 12 - 1) as usize],
            "origin {o}"
        );
    }
}

#[test]
fn cape_cod_on_a_diagonal_trends_to_the_fitted_columns_valuation() {
    // Premium for 2021 at 24 months with no paid there: the triangle is
    // valued at December 2022, the paid diagonal at December 2021.
    let origin = [2020, 2020, 2021, 2021].map(Month::january);
    let book = Triangle::from_long(&Long {
        keys: &[],
        origin: &origin,
        development: DevelopmentColumn::Age(&[12, 24, 12, 24]),
        values: &[
            ("paid", &[100.0, 150.0, 200.0, f64::NAN]),
            ("premium", &[400.0, 450.0, 500.0, 520.0]),
        ],
        origin_grain: Grain::Year,
        development_grain: Grain::Year,
        cumulative: true,
    })
    .unwrap();
    assert_eq!(book.valuation(), Month::new(2022, 12).unwrap());
    let cc = CapeCod {
        trend: 0.1,
        decay: 0.75,
        chain_ladder: ChainLadder::default(),
    };
    let on_triangle = cc.fit(&book, "paid", "premium").unwrap();
    let frozen = CapeCod {
        chain_ladder: on_triangle.expected_loss.chain_ladder.to_chain_ladder(),
        ..cc
    };
    let d = LatestDiagonal::from_triangle(&book, "paid", Some("premium")).unwrap();
    assert_eq!(d.exposure(), Some([450.0, 520.0].as_slice()));
    let on_diagonal = frozen.fit_diagonal(&d).unwrap();
    // One year less of trend on every origin: the trended aprioris differ by
    // 1.1, the detrended aprioris and the ultimates agree.
    let (t, g) = (&on_triangle.expected_loss, &on_diagonal.expected_loss);
    assert_close(&g.ultimate, &t.ultimate, 1e-13, "Cape Cod ultimates");
    assert_close(&g.apriori, &t.apriori, 1e-13, "Cape Cod aprioris");
    let lifted: Vec<f64> = on_diagonal
        .trended_apriori
        .iter()
        .map(|a| a * 1.1)
        .collect();
    assert_close(
        &lifted,
        &on_triangle.trended_apriori,
        1e-13,
        "trended aprioris",
    );
}

#[test]
fn the_applied_fit_selects_the_sources_factors() {
    let tri = triangle("genins");
    let source = ChainLadder {
        tail: Tail::from(1.05),
        ..Default::default()
    }
    .fit(&tri, "values")
    .unwrap();
    let d = LatestDiagonal::from_triangle(&tri, "values", None).unwrap();
    let fit = source.to_chain_ladder().fit_diagonal(&d).unwrap();
    let want: Vec<Option<f64>> = source.ldf().iter().copied().map(Some).collect();
    assert_eq!(fit.development.selected, want);
    assert!(fit.development.weight.iter().all(|&w| w == 1.0));
    // A diagonal has no link ratios, so the estimate is NaN.
    assert!(fit.development.ldf.iter().all(|f| f.is_nan()));
}

// D. Errors.

#[test]
fn an_age_outside_the_sources_is_an_error() {
    let diag = |reason: &str, e: Error| match e {
        Error::Diagonal(why) => assert!(why.contains(reason), "{why}"),
        other => panic!("{other} is not a latest-diagonal error"),
    };
    let full = triangle("genins");
    let d = LatestDiagonal::from_triangle(&full, "values", None).unwrap();
    // A source at most 60 months old.
    let young = ChainLadder::default()
        .fit(&target("genins", "from_2006"), "values")
        .unwrap();
    diag(
        "origin 2001 is at 120 months, outside the pattern's ages from 12 to 60 months",
        young.to_chain_ladder().fit_diagonal(&d).unwrap_err(),
    );
    // A source from 24 months.
    let late = ChainLadder::default()
        .fit(&remake(&full, 0, |_, age| age >= 24, |_, _, v| v), "values")
        .unwrap();
    diag(
        "origin 2010 is at 12 months, outside the pattern's ages from 24 to 120 months",
        late.to_chain_ladder().fit_diagonal(&d).unwrap_err(),
    );
}

#[test]
fn from_triangle_names_what_is_wrong() {
    let origin = [2020, 2020, 2021, 2020, 2020, 2021].map(Month::january);
    let keyed = Triangle::from_long(&Long {
        keys: &[("lob", &["a", "a", "a", "b", "b", "b"])],
        origin: &origin,
        development: DevelopmentColumn::Age(&[12, 24, 12, 12, 24, 12]),
        values: &[("paid", &[100.0, 150.0, 200.0, 10.0, 15.0, 20.0])],
        origin_grain: Grain::Year,
        development_grain: Grain::Year,
        cumulative: true,
    })
    .unwrap();
    assert_eq!(
        LatestDiagonal::from_triangle(&keyed, "paid", None).unwrap_err(),
        Error::MultipleSegments(2)
    );
    let tri = triangle_columns("clrd_wkcomp", &["paid", "premium"]);
    assert!(matches!(
        LatestDiagonal::from_triangle(&tri, "incurred", None).unwrap_err(),
        Error::UnknownColumn(_)
    ));
    assert!(matches!(
        LatestDiagonal::from_triangle(&tri, "paid", Some("exposure")).unwrap_err(),
        Error::UnknownColumn(_)
    ));
    // 2021 has no cell between 2020 and 2022.
    let gap = Triangle::from_long(&Long {
        keys: &[],
        origin: &[2020, 2020, 2022].map(Month::january),
        development: DevelopmentColumn::Age(&[12, 24, 12]),
        values: &[("paid", &[100.0, 150.0, 200.0])],
        origin_grain: Grain::Year,
        development_grain: Grain::Year,
        cumulative: true,
    })
    .unwrap();
    assert!(matches!(
        LatestDiagonal::from_triangle(&gap, "paid", None).unwrap_err(),
        Error::EmptyOrigin(_)
    ));
    // A premium of 0 for 1990.
    let zero = remake(
        &tri,
        0,
        |_, _| true,
        |c, year, v| {
            if c == 1 && year == 1990 { 0.0 } else { v }
        },
    );
    match LatestDiagonal::from_triangle(&zero, "paid", Some("premium")).unwrap_err() {
        Error::InvalidExposure { column, origin } => {
            assert_eq!((column.as_str(), origin.as_str()), ("premium", "1990"))
        }
        other => panic!("{other} is not an exposure error"),
    }
}

#[test]
fn the_expected_loss_methods_need_the_exposure() {
    let tri = triangle_columns("clrd_wkcomp", &["paid", "premium"]);
    let source = ChainLadder::default().fit(&tri, "paid").unwrap();
    let d = LatestDiagonal::from_triangle(&tri, "paid", None).unwrap();
    let e = BornhuetterFerguson {
        apriori: 0.6.into(),
        chain_ladder: source.to_chain_ladder(),
    }
    .fit_diagonal(&d)
    .unwrap_err();
    match e {
        Error::Diagonal(why) => assert!(why.contains("need an exposure"), "{why}"),
        other => panic!("{other} is not a latest-diagonal error"),
    }
}

#[test]
fn a_carried_tail_is_spread_down_to_its_floor() {
    // The carried tail is spread at decay 0.5, where a factor below
    // exp(-0.75), about 0.4724, cannot be (chainladder-python's seed has no
    // real root there); the source spreads each at 0.9.
    let tri = triangle("genins");
    let d = LatestDiagonal::from_triangle(&tri, "values", None).unwrap();
    let floor = (-0.75f64).exp();
    for factor in [0.45, 0.4723, 0.4724, 0.48] {
        let source = ChainLadder {
            tail: TailConstant {
                factor,
                decay: 0.9,
                attachment_age: None,
            }
            .into(),
            ..Default::default()
        }
        .fit(&tri, "values")
        .unwrap();
        assert!((source.tail.factor - factor).abs() < 1e-12);
        let applied = source.to_chain_ladder().fit_diagonal(&d);
        if factor < floor {
            match applied.unwrap_err() {
                Error::Tail(why) => assert!(why.contains("too small to spread"), "{why}"),
                other => panic!("{other} is not a tail error"),
            }
        } else {
            let fit = applied.unwrap();
            assert!(
                close_to(fit.tail.factor, source.tail.factor, 1e-14),
                "tail {factor}: {}",
                fit.tail.factor
            );
            assert_close(&fit.ultimate, &source.ultimate, 1e-14, "above the floor");
        }
    }
}

#[test]
fn a_source_with_one_age_has_no_factors() {
    let one = Triangle::from_long(&Long {
        keys: &[],
        origin: &[2020, 2021].map(Month::january),
        development: DevelopmentColumn::Age(&[12, 12]),
        values: &[("paid", &[100.0, 200.0])],
        origin_grain: Grain::Year,
        development_grain: Grain::Year,
        cumulative: true,
    })
    .unwrap();
    let source = ChainLadder::default().fit(&one, "paid").unwrap();
    let frozen = source.to_chain_ladder();
    assert!(
        frozen
            .development
            .pattern
            .as_ref()
            .unwrap()
            .values
            .is_empty()
    );
    let d = LatestDiagonal::from_triangle(&one, "paid", None).unwrap();
    assert_eq!(
        frozen.fit_diagonal(&d).unwrap_err(),
        Error::Pattern {
            age: None,
            reason: "the pattern has no values",
        }
    );
}
