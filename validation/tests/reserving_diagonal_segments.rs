//! Latest diagonals over segments (decision 19 of
//! `docs/design/reserving-v02.md`, issue #544): `LatestDiagonals`, the
//! methods' `fit_diagonal_segments` and `fit_diagonal_segments_from`, and
//! `LatestDiagonals::developments` with its `SegmentMap`.
//!
//! * GenIns and a perturbed GenIns as two lines of one keyed triangle,
//!   fitted with `fit_segments` and a curve tail, applied to the keyed
//!   triangle's diagonals by label: each segment projects as its source
//!   fitted (factors to the bit, tail, factors to ultimate and ultimates
//!   within 1e-14). A map of b onto a, and a single-segment source
//!   broadcast, equal the per-segment `fit_diagonal` to the bit.
//! * The expected-loss methods on the CAS workers' compensation triangle
//!   twice: each segment equal to the per-segment `fit_diagonal` to the bit
//!   when every segment shares one valuation, and Cape Cod with one line
//!   valued a year earlier trended to the common valuation.
//! * `from_long` with keys and valuations equals `from_triangle`.
//! * Errors: every map error, a method with a pattern or tail given with a
//!   development, an origin given twice in a segment and the row of a
//!   non-positive age.

use prospicio_reserving::{
    Benktander, BornhuetterFerguson, CapeCod, ChainLadder, ChainLadderFit, Development,
    DevelopmentColumn, DiagonalAge, Error, ExpectedLoss, Grain, Label, Lag, LatestDiagonal,
    LatestDiagonals, LongDiagonal, Month, PatternKind, SegmentFits, SegmentMap, SelectedPattern,
    Tail, TailCurve, Triangle,
};
use prospicio_validation::{triangle, triangle_columns};

/// Equal to the bit: the same `Debug` text, which prints every float in
/// full and a NaN (a diagonal's estimated factors) as NaN.
macro_rules! assert_bits {
    ($a:expr, $b:expr $(,)?) => {
        assert_eq!(format!("{:?}", $a), format!("{:?}", $b))
    };
}

/// A copy of the single-segment `tri`, its values mapped by `value` (by
/// column position, origin year and value) and its origins `shift` years
/// later.
fn remake(tri: &Triangle, shift: i32, value: impl Fn(usize, i32, f64) -> f64) -> Triangle {
    let (origin, age, values) = cells(tri, shift, value);
    let columns = tri.columns().to_vec();
    let values: Vec<(&str, &[f64])> = columns
        .iter()
        .zip(&values)
        .map(|(n, v)| (n.as_str(), v.as_slice()))
        .collect();
    Triangle::from_long(&prospicio_reserving::Long {
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

/// The observed cells of a single-segment triangle as long columns.
type Cells = (Vec<Month>, Vec<Lag>, Vec<Vec<f64>>);

fn cells(tri: &Triangle, shift: i32, value: impl Fn(usize, i32, f64) -> f64) -> Cells {
    let n_columns = tri.columns().len();
    let (mut origin, mut age, mut values) = (Vec::new(), Vec::new(), vec![Vec::new(); n_columns]);
    for (o, p) in tri.origins().iter().enumerate() {
        let year = p.start().year();
        for (d, &a) in tri.development().iter().enumerate() {
            let row: Option<Vec<f64>> = (0..n_columns).map(|c| tri.get(0, c, o, d)).collect();
            let Some(row) = row else { continue };
            origin.push(Month::january(year + shift));
            age.push(a);
            for (c, v) in row.into_iter().enumerate() {
                values[c].push(value(c, year, v));
            }
        }
    }
    (origin, age, values)
}

/// The single-segment `books` as the lines of one triangle keyed by `lob`.
fn keyed(books: &[(&str, &Triangle)]) -> Triangle {
    let columns = books[0].1.columns().to_vec();
    let (mut lob, mut origin, mut age) = (Vec::new(), Vec::new(), Vec::new());
    let mut values = vec![Vec::new(); columns.len()];
    for (label, tri) in books {
        let (o, a, v) = cells(tri, 0, |_, _, v| v);
        lob.extend(std::iter::repeat_n(*label, o.len()));
        origin.extend(o);
        age.extend(a);
        for (c, column) in v.into_iter().enumerate() {
            values[c].extend(column);
        }
    }
    let values: Vec<(&str, &[f64])> = columns
        .iter()
        .zip(&values)
        .map(|(n, v)| (n.as_str(), v.as_slice()))
        .collect();
    Triangle::from_long(&prospicio_reserving::Long {
        keys: &[("lob", lob.as_slice())],
        origin: &origin,
        development: DevelopmentColumn::Age(&age),
        values: &values,
        origin_grain: Grain::Year,
        development_grain: Grain::Year,
        cumulative: true,
    })
    .unwrap()
}

fn genins_perturbed() -> Triangle {
    remake(&triangle("genins"), 0, |_, year, v| {
        v * (1.0 + 0.01 * f64::from(year - 2000))
    })
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

/// As `reserving_fit_apply.rs` checks a fit applied to its own book.
fn assert_same_projection(applied: &ChainLadderFit, source: &ChainLadderFit, what: &str) {
    assert_eq!(applied.origins, source.origins, "{what}: origins");
    assert_eq!(applied.ldf(), source.ldf(), "{what}: factors");
    assert!(
        close_to(applied.tail.factor, source.tail.factor, 1e-14),
        "{what}: tail {} against {}",
        applied.tail.factor,
        source.tail.factor
    );
    assert_close(&applied.cdf, &source.cdf, 1e-14, what);
    assert_close(&applied.ultimate, &source.ultimate, 1e-14, what);
}

/// GenIns as line a and the perturbed GenIns as line b, and their fits with
/// a curve tail.
fn genins_lines() -> (Triangle, SegmentFits<ChainLadderFit>) {
    let (a, b) = (triangle("genins"), genins_perturbed());
    let tri = keyed(&[("a", &a), ("b", &b)]);
    let fits = ChainLadder {
        tail: TailCurve::default().into(),
        ..Default::default()
    }
    .fit_segments(&tri, "values")
    .unwrap();
    assert_ne!(fits.fits[0].ldf(), fits.fits[1].ldf());
    (tri, fits)
}

fn label(s: &str) -> Label {
    Label::from(s)
}

#[test]
fn each_segment_takes_the_source_segment_of_its_label() {
    let (tri, fits) = genins_lines();
    let diagonals = LatestDiagonals::from_triangle(&tri, "values", None).unwrap();
    assert_eq!(diagonals.labels(), [label("a"), label("b")]);
    assert_eq!(diagonals.key_names(), ["lob"]);
    let developments = diagonals.developments(&fits, &SegmentMap::ByLabel).unwrap();
    let applied = ChainLadder::default()
        .fit_diagonal_segments_from(&diagonals, &developments)
        .unwrap();
    assert_eq!(applied.labels, fits.labels);
    for (k, source) in fits.fits.iter().enumerate() {
        assert_same_projection(&applied.fits[k], source, &format!("segment {k}"));
    }
    // The long tables line up with the source's.
    assert_eq!(applied.to_long().keys, fits.to_long().keys);
}

#[test]
fn a_map_of_pairs_develops_a_segment_with_another() {
    let (tri, fits) = genins_lines();
    let diagonals = LatestDiagonals::from_triangle(&tri, "values", None).unwrap();
    let map = SegmentMap::Pairs(vec![(label("b"), label("a")), (label("a"), label("a"))]);
    let developments = diagonals.developments(&fits, &map).unwrap();
    let applied = ChainLadder::default()
        .fit_diagonal_segments_from(&diagonals, &developments)
        .unwrap();
    let a = fits.get(&label("a")).unwrap().to_chain_ladder();
    let b_diagonal = diagonals.get(&label("b")).unwrap();
    assert_bits!(applied.fits[1], a.fit_diagonal(b_diagonal).unwrap());
    let a_diagonal = diagonals.get(&label("a")).unwrap();
    assert_bits!(applied.fits[0], a.fit_diagonal(a_diagonal).unwrap());
}

#[test]
fn a_single_segment_source_develops_every_segment() {
    let (tri, _) = genins_lines();
    let source = ChainLadder {
        tail: Tail::from(1.05),
        ..Default::default()
    }
    .fit_segments(&triangle("genins"), "values")
    .unwrap();
    assert!(source.key_names.is_empty());
    let diagonals = LatestDiagonals::from_triangle(&tri, "values", None).unwrap();
    let developments = diagonals
        .developments(&source, &SegmentMap::ByLabel)
        .unwrap();
    let applied = ChainLadder::default()
        .fit_diagonal_segments_from(&diagonals, &developments)
        .unwrap();
    let frozen = source.fits[0].to_chain_ladder();
    for (k, (_, d)) in diagonals.iter().enumerate() {
        assert_bits!(applied.fits[k], frozen.fit_diagonal(d).unwrap());
    }
    // A single segment without keys is LatestDiagonal's fit itself.
    let one = LatestDiagonal::from_triangle(&triangle("genins"), "values", None).unwrap();
    let fits = frozen
        .fit_diagonal_segments(&LatestDiagonals::from(one.clone()))
        .unwrap();
    assert_eq!(fits.labels, [Label::default()]);
    assert_bits!(fits.fits, [frozen.fit_diagonal(&one).unwrap()]);
}

/// The workers' compensation triangle as lines a and b, b's paid and
/// premium mapped by `value` and its origins `shift` years later.
fn wkcomp_lines(shift: i32, value: impl Fn(usize, i32, f64) -> f64) -> Triangle {
    let a = triangle_columns("clrd_wkcomp", &["paid", "premium"]);
    let b = remake(&a, shift, value);
    keyed(&[("a", &a), ("b", &b)])
}

/// The workers' compensation chain ladder's factors to ultimate.
fn wkcomp_pattern() -> ChainLadder {
    let tri = triangle_columns("clrd_wkcomp", &["paid", "premium"]);
    let own = ChainLadder::default().fit(&tri, "paid").unwrap();
    ChainLadder {
        development: Development {
            pattern: Some(SelectedPattern::new(
                PatternKind::AgeToUltimate,
                tri.development()
                    .iter()
                    .copied()
                    .zip(own.cdf.iter().copied())
                    .collect(),
            )),
            ..Default::default()
        },
        ..Default::default()
    }
}

#[test]
fn the_expected_loss_methods_fit_each_segment_as_alone() {
    let tri = wkcomp_lines(0, |c, year, v| {
        let v = v * (1.0 + 0.01 * f64::from(year - 1987));
        if c == 1 { v * 1.1 } else { v }
    });
    let diagonals = LatestDiagonals::from_triangle(&tri, "paid", Some("premium")).unwrap();
    let chain_ladder = wkcomp_pattern();
    let el = ExpectedLoss {
        apriori: 0.6.into(),
        chain_ladder: chain_ladder.clone(),
    };
    let bf = BornhuetterFerguson {
        apriori: 0.6.into(),
        chain_ladder: chain_ladder.clone(),
    };
    let bk = Benktander {
        apriori: 0.6.into(),
        n_iters: 3,
        chain_ladder: chain_ladder.clone(),
    };
    let cc = CapeCod {
        trend: 0.02,
        decay: 0.75,
        chain_ladder: chain_ladder.clone(),
    };
    let el_fits = el.fit_diagonal_segments(&diagonals).unwrap();
    let bf_fits = bf.fit_diagonal_segments(&diagonals).unwrap();
    let bk_fits = bk.fit_diagonal_segments(&diagonals).unwrap();
    let cc_fits = cc.fit_diagonal_segments(&diagonals).unwrap();
    for (k, (_, d)) in diagonals.iter().enumerate() {
        assert_bits!(el_fits.fits[k], el.fit_diagonal(d).unwrap());
        assert_bits!(bf_fits.fits[k], bf.fit_diagonal(d).unwrap());
        assert_bits!(bk_fits.fits[k], bk.fit_diagonal(d).unwrap());
        assert_bits!(cc_fits.fits[k], cc.fit_diagonal(d).unwrap());
    }
    // The segments differ, so each one has its own fit.
    assert_ne!(bf_fits.fits[0].ultimate, bf_fits.fits[1].ultimate);

    // With the development from a source fit of each line, the methods keep
    // their own apriori, n_iters, trend and decay.
    let source = ChainLadder::default().fit_segments(&tri, "paid").unwrap();
    let developments = diagonals
        .developments(&source, &SegmentMap::ByLabel)
        .unwrap();
    let unset = |apriori: f64| BornhuetterFerguson {
        apriori: apriori.into(),
        chain_ladder: ChainLadder::default(),
    };
    let from = unset(0.6)
        .fit_diagonal_segments_from(&diagonals, &developments)
        .unwrap();
    for (k, (_, d)) in diagonals.iter().enumerate() {
        let own = BornhuetterFerguson {
            apriori: 0.6.into(),
            chain_ladder: source.fits[k].to_chain_ladder(),
        };
        assert_bits!(from.fits[k], own.fit_diagonal(d).unwrap());
    }
    let from = CapeCod {
        trend: 0.02,
        decay: 0.75,
        chain_ladder: ChainLadder::default(),
    }
    .fit_diagonal_segments_from(&diagonals, &developments)
    .unwrap();
    for (k, (_, d)) in diagonals.iter().enumerate() {
        let own = CapeCod {
            trend: 0.02,
            decay: 0.75,
            chain_ladder: source.fits[k].to_chain_ladder(),
        };
        assert_bits!(from.fits[k], own.fit_diagonal(d).unwrap());
    }
    let from = ExpectedLoss {
        apriori: 0.6.into(),
        chain_ladder: ChainLadder::default(),
    }
    .fit_diagonal_segments_from(&diagonals, &developments)
    .unwrap();
    assert_eq!(from.len(), 2);
    let from = Benktander {
        apriori: 0.6.into(),
        n_iters: 3,
        chain_ladder: ChainLadder::default(),
    }
    .fit_diagonal_segments_from(&diagonals, &developments)
    .unwrap();
    assert_eq!(from.len(), 2);
}

#[test]
fn cape_cod_trends_every_segment_to_one_valuation() {
    // Line b is valued a year before line a.
    let tri = wkcomp_lines(-1, |_, _, v| v);
    let diagonals = LatestDiagonals::from_triangle(&tri, "paid", Some("premium")).unwrap();
    let (a, b) = (&diagonals.diagonals()[0], &diagonals.diagonals()[1]);
    assert_eq!(b.valuation().months_since(a.valuation()), -12);
    assert_eq!(diagonals.valuation(), a.valuation());
    let trend = 0.05;
    let cc = CapeCod {
        trend,
        decay: 0.75,
        chain_ladder: wkcomp_pattern(),
    };
    let fits = cc.fit_diagonal_segments(&diagonals).unwrap();
    assert_bits!(fits.fits[0], cc.fit_diagonal(a).unwrap());
    let own = cc.fit_diagonal(b).unwrap();
    let common = &fits.fits[1];
    let lifted: Vec<f64> = own
        .trended_apriori
        .iter()
        .map(|t| t * (1.0 + trend))
        .collect();
    assert_close(&common.trended_apriori, &lifted, 1e-12, "trended apriori");
    assert_close(
        &common.expected_loss.apriori,
        &own.expected_loss.apriori,
        1e-12,
        "detrended apriori",
    );
    assert_close(
        &common.expected_loss.ultimate,
        &own.expected_loss.ultimate,
        1e-12,
        "ultimates",
    );
}

#[test]
fn a_long_table_with_keys_and_valuations_is_the_triangle_diagonal() {
    let tri = wkcomp_lines(0, |_, _, v| 2.0 * v);
    let from_triangle = LatestDiagonals::from_triangle(&tri, "paid", Some("premium")).unwrap();
    // Each segment's rows, newest origin first and line b first, with each
    // row's valuation.
    let (mut lob, mut origin, mut valuation, mut latest, mut exposure) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for (l, d) in from_triangle
        .labels()
        .iter()
        .zip(from_triangle.diagonals())
        .rev()
    {
        for k in (0..d.origins().len()).rev() {
            lob.push(l.parts()[0].clone());
            origin.push(d.origins()[k]);
            valuation.push(d.origins()[k].add_months(i64::from(d.ages()[k]) - 1));
            latest.push(d.latest()[k]);
            exposure.push(d.exposure().unwrap()[k]);
        }
    }
    let lob: Vec<&str> = lob.iter().map(String::as_str).collect();
    let long = LatestDiagonals::from_long(&LongDiagonal {
        keys: &[("lob", &lob)],
        origin: &origin,
        age: DiagonalAge::Valuations(&valuation),
        latest: &latest,
        exposure: Some(&exposure),
        origin_grain: Grain::Year,
    })
    .unwrap();
    assert_eq!(long.labels(), from_triangle.labels());
    let pattern = wkcomp_pattern();
    let bf = BornhuetterFerguson {
        apriori: 0.6.into(),
        chain_ladder: pattern,
    };
    assert_bits!(
        bf.fit_diagonal_segments(&long).unwrap(),
        bf.fit_diagonal_segments(&from_triangle).unwrap()
    );
    // with_exposure aligns to the rows segment by segment.
    let n: usize = long.diagonals().iter().map(|d| d.origins().len()).sum();
    let given: Vec<f64> = (0..n).map(|k| 1000.0 + k as f64).collect();
    let set = long.clone().with_exposure(&given).unwrap();
    let first = long.diagonals()[0].origins().len();
    assert_eq!(set.diagonals()[0].exposure().unwrap(), &given[..first]);
    assert_eq!(set.diagonals()[1].exposure().unwrap(), &given[first..]);
}

#[test]
fn the_map_names_what_is_wrong() {
    let (tri, fits) = genins_lines();
    let diagonals = LatestDiagonals::from_triangle(&tri, "values", None).unwrap();
    let message = |map: SegmentMap, source: &SegmentFits<ChainLadderFit>| match diagonals
        .developments(source, &map)
    {
        Err(Error::Diagonal(m)) => m,
        other => panic!("expected a diagonal error, got {other:?}"),
    };
    // A target with no source of its label.
    let only_a = fits.segment(&[("lob", "a")]).unwrap();
    let mut a_and_c = fits.clone();
    a_and_c.labels[1] = label("c");
    let m = message(SegmentMap::ByLabel, &a_and_c);
    assert!(
        m.contains("segment b of the diagonals has no segment"),
        "{m}"
    );
    // Other keys.
    let mut other_keys = fits.clone();
    other_keys.key_names = vec!["state".into()];
    let m = message(SegmentMap::ByLabel, &other_keys);
    assert!(
        m.contains("the source has keys [state] and the diagonals [lob]"),
        "{m}"
    );
    // A one-segment source broadcasts even with keys.
    assert!(
        diagonals
            .developments(&only_a, &SegmentMap::ByLabel)
            .is_ok()
    );
    let pairs = |p: &[(&str, &str)]| {
        SegmentMap::Pairs(p.iter().map(|(t, s)| (label(t), label(s))).collect())
    };
    let m = message(pairs(&[("a", "a"), ("a", "b"), ("b", "a")]), &fits);
    assert!(m.contains("lists the target a twice"), "{m}");
    let m = message(pairs(&[("a", "a")]), &fits);
    assert!(
        m.contains("gives segment b of the diagonals no source"),
        "{m}"
    );
    let m = message(pairs(&[("a", "a"), ("c", "a")]), &fits);
    assert!(
        m.contains("target c is not a segment of the diagonals"),
        "{m}"
    );
    let m = message(pairs(&[("a", "a"), ("b", "c")]), &fits);
    assert!(m.contains("source c is not a segment of the source"), "{m}");
}

#[test]
fn a_development_replaces_the_methods_own() {
    let (tri, fits) = genins_lines();
    let diagonals = LatestDiagonals::from_triangle(&tri, "values", None).unwrap();
    let developments = diagonals.developments(&fits, &SegmentMap::ByLabel).unwrap();
    let message = |cl: ChainLadder| match cl.fit_diagonal_segments_from(&diagonals, &developments) {
        Err(Error::Diagonal(m)) => m,
        other => panic!("expected a diagonal error, got {other:?}"),
    };
    let tailed = ChainLadder {
        tail: Tail::from(1.05),
        ..Default::default()
    };
    assert!(message(tailed).contains("the development comes from the source fit"));
    let patterned = ChainLadder {
        development: Development {
            pattern: Some(SelectedPattern::new(PatternKind::AgeToAge, vec![(12, 2.0)])),
            ..Default::default()
        },
        ..Default::default()
    };
    assert!(message(patterned).contains("give the method no pattern"));
    // Link ratio exclusions would be ignored silently, so they are refused
    // too (decision 19 item 5).
    let latest = ChainLadder {
        development: Development {
            n_periods: Some(3),
            ..Default::default()
        },
        ..Default::default()
    };
    assert!(message(latest).contains("no link ratio exclusions"));
    let dropped = ChainLadder {
        development: Development {
            drop_high: 1,
            ..Default::default()
        },
        ..Default::default()
    };
    assert!(message(dropped).contains("no link ratio exclusions"));
    let dropped_low = ChainLadder {
        development: Development {
            drop_low: 1,
            ..Default::default()
        },
        ..Default::default()
    };
    assert!(message(dropped_low).contains("no link ratio exclusions"));
    // A tail of exactly 1 is no tail.
    let one = ChainLadder {
        tail: Tail::from(1.0),
        ..Default::default()
    };
    assert!(
        one.fit_diagonal_segments_from(&diagonals, &developments)
            .is_ok()
    );
    // The developments must be keyed by the diagonals' segments.
    let one_segment = diagonals.segment(&[("lob", "a")]).unwrap();
    let m = match ChainLadder::default().fit_diagonal_segments_from(&one_segment, &developments) {
        Err(Error::Diagonal(m)) => m,
        other => panic!("expected a diagonal error, got {other:?}"),
    };
    assert!(
        m.contains("the developments' segments are not the diagonals'"),
        "{m}"
    );
}

#[test]
fn a_long_table_reports_its_rows_and_segments() {
    let origin = [2023, 2024, 2023, 2023].map(Month::january);
    let lob = ["a", "a", "b", "b"];
    let long = |ages: &[Lag]| {
        LatestDiagonals::from_long(&LongDiagonal {
            keys: &[("lob", &lob)],
            origin: &origin,
            age: DiagonalAge::Ages(ages),
            latest: &[1.0, 2.0, 3.0, 4.0],
            exposure: None,
            origin_grain: Grain::Year,
        })
    };
    // Line b gives 2023 twice: the error names the segment.
    match long(&[24, 12, 24, 24]) {
        Err(Error::InSegment { label, source }) => {
            assert_eq!(label, "b");
            assert!(matches!(*source, Error::Diagonal(ref m) if m.contains("given twice")));
        }
        other => panic!("expected a segment error, got {other:?}"),
    }
    // The row of a zero age is the table's.
    assert_eq!(
        long(&[24, 12, 24, 0]),
        Err(Error::NonPositiveAge { row: 3 })
    );
    // A valuation before an origin starts, at the table's row too.
    let early = LatestDiagonals::from_long(&LongDiagonal {
        keys: &[("lob", &lob)],
        origin: &origin,
        age: DiagonalAge::Valuation(Month::new(2023, 12).unwrap()),
        latest: &[1.0, 2.0, 3.0, 4.0],
        exposure: None,
        origin_grain: Grain::Year,
    });
    assert_eq!(early, Err(Error::NonPositiveAge { row: 1 }));
    let repeated = LatestDiagonals::from_long(&LongDiagonal {
        keys: &[("lob", &lob), ("lob", &lob)],
        origin: &origin,
        age: DiagonalAge::Ages(&[24, 12, 24, 12]),
        latest: &[1.0, 2.0, 3.0, 4.0],
        exposure: None,
        origin_grain: Grain::Year,
    });
    assert_eq!(repeated, Err(Error::DuplicateKey("lob".into())));
    let short = LatestDiagonals::from_long(&LongDiagonal {
        keys: &[("lob", &lob[..3])],
        origin: &origin,
        age: DiagonalAge::Ages(&[24, 12, 24, 12]),
        latest: &[1.0, 2.0, 3.0, 4.0],
        exposure: None,
        origin_grain: Grain::Year,
    });
    assert!(matches!(short, Err(Error::LengthMismatch { .. })));
}
