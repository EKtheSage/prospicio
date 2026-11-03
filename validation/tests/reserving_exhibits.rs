//! Exhibits and the weighted selection of ultimates
//! (`docs/design/reserving-v02.md`, decision 30; issue #557).
//!
//! * Friedland (2010), *Estimating Unpaid Claims Using Basic Techniques*:
//!   chapter 7, Exhibit I, sheets 1 to 4 (U.S. Industry Auto,
//!   `data/friedland_us_auto.csv`), the averages, selected factors,
//!   factors to ultimate and percents developed to the printed digits and
//!   the ultimates, IBNR and unpaid to the unit; chapter 15, Exhibit I,
//!   sheets 2 and 6 (XYZ Insurer), the selected ultimate and IBNR from the
//!   printed projections of each method, within the book's rounding.
//! * chainladder-python's `VotingChainladder` on GenIns with premium, every
//!   row of `reference/reserving_exhibits_python.csv`
//!   (`scripts/reserving_exhibits_python.py`).

use prospicio_reserving::{
    Apriori, BornhuetterFerguson, CapeCod, ChainLadder, Development, DevelopmentExhibit,
    ExpectedLoss, MethodWeights, PatternKind, Period, SelectedPattern, Tail, UltimateExhibit,
    UltimateSelection, friedland_averages,
};
use prospicio_validation::{Case, check, reference, triangle_columns};

const US_YEARS: std::ops::RangeInclusive<i32> = 1998..=2007;
const AGES: [u32; 10] = [12, 24, 36, 48, 60, 72, 84, 96, 108, 120];

fn us() -> prospicio_reserving::Triangle {
    triangle_columns("friedland_us_auto", &["reported", "paid", "premium"])
}

/// Each value equals the printed one at `digits` decimals.
fn assert_printed(what: &str, got: &[f64], book: &[f64], digits: i32) {
    assert_eq!(got.len(), book.len(), "{what}: lengths");
    let scale = 10f64.powi(digits);
    for (k, (g, b)) in got.iter().zip(book).enumerate() {
        assert!(
            ((g * scale).round() - b * scale).abs() < 1e-6,
            "{what}[{k}]: {g} prints as {}, book {b}",
            (g * scale).round() / scale
        );
    }
}

fn assert_close(what: &str, got: &[f64], book: &[f64], tol: f64) {
    assert_eq!(got.len(), book.len(), "{what}: lengths");
    for (k, (g, b)) in got.iter().zip(book).enumerate() {
        assert!((g - b).abs() <= tol, "{what}[{k}]: {g} against {b}");
    }
}

fn pattern(kind: PatternKind, values: &[f64]) -> Development {
    Development {
        pattern: Some(SelectedPattern::new(
            kind,
            AGES.iter().copied().zip(values.iter().copied()).collect(),
        )),
        ..Default::default()
    }
}

/// Chapter 7, Exhibit I, sheets 1 (reported) and 2 (paid), parts 2 to 4.
#[test]
fn friedland_development_exhibits() {
    let tri = us();
    // Part 3, in friedland_averages' order: simple latest 5 and 3, medial
    // latest 5x1, volume-weighted latest 5 and 3, geometric latest 4.
    let book_averages: [(&str, [[f64; 9]; 6]); 2] = [
        (
            "reported",
            [
                [
                    1.168, 1.058, 1.027, 1.011, 1.004, 1.003, 1.002, 1.001, 1.000,
                ],
                [
                    1.164, 1.056, 1.027, 1.012, 1.005, 1.003, 1.002, 1.001, 1.000,
                ],
                [
                    1.165, 1.057, 1.027, 1.010, 1.004, 1.003, 1.002, 1.001, 1.000,
                ],
                [
                    1.168, 1.058, 1.027, 1.011, 1.004, 1.003, 1.002, 1.001, 1.000,
                ],
                [
                    1.164, 1.056, 1.027, 1.012, 1.005, 1.003, 1.002, 1.001, 1.000,
                ],
                [
                    1.164, 1.057, 1.027, 1.011, 1.004, 1.003, 1.002, 1.001, 1.000,
                ],
            ],
        ),
        (
            "paid",
            [
                [
                    1.712, 1.189, 1.091, 1.044, 1.019, 1.010, 1.005, 1.002, 1.002,
                ],
                [
                    1.702, 1.186, 1.091, 1.044, 1.019, 1.009, 1.005, 1.002, 1.002,
                ],
                [
                    1.708, 1.188, 1.091, 1.044, 1.019, 1.010, 1.005, 1.002, 1.002,
                ],
                [
                    1.712, 1.189, 1.091, 1.044, 1.019, 1.010, 1.005, 1.002, 1.002,
                ],
                [
                    1.702, 1.186, 1.091, 1.044, 1.019, 1.009, 1.005, 1.002, 1.002,
                ],
                [
                    1.706, 1.188, 1.091, 1.044, 1.019, 1.010, 1.005, 1.002, 1.002,
                ],
            ],
        ),
    ];
    // Part 2: the 12-24 column and the 1998 row.
    let book_first_link = [
        [
            1.166, 1.182, 1.200, 1.193, 1.184, 1.162, 1.159, 1.160, 1.173,
        ],
        [
            1.792, 1.768, 1.762, 1.744, 1.735, 1.719, 1.703, 1.701, 1.703,
        ],
    ];
    let book_1998 = [
        [
            1.166, 1.056, 1.027, 1.012, 1.004, 1.002, 1.001, 1.001, 1.000,
        ],
        [
            1.792, 1.206, 1.096, 1.046, 1.019, 1.010, 1.005, 1.002, 1.002,
        ],
    ];
    // Part 4: the selected factors (the latest-3 volume-weighted averages,
    // rounded), the tail, the factors to ultimate and the percents.
    let selected = [
        [
            1.164, 1.056, 1.027, 1.012, 1.005, 1.003, 1.002, 1.001, 1.000,
        ],
        [
            1.702, 1.186, 1.091, 1.044, 1.019, 1.009, 1.005, 1.002, 1.002,
        ],
    ];
    let tails = [1.000, 1.002];
    let book_cdf = [
        [
            1.292, 1.110, 1.051, 1.023, 1.011, 1.006, 1.003, 1.001, 1.000, 1.000,
        ],
        [
            2.390, 1.404, 1.184, 1.085, 1.040, 1.020, 1.011, 1.006, 1.004, 1.002,
        ],
    ];
    let book_percent = [
        [77.4, 90.1, 95.1, 97.8, 98.9, 99.4, 99.7, 99.9, 100.0, 100.0],
        [41.8, 71.2, 84.5, 92.2, 96.2, 98.0, 98.9, 99.4, 99.6, 99.8],
    ];
    for (c, (column, book)) in book_averages.iter().enumerate() {
        let selection = ChainLadder {
            development: pattern(PatternKind::AgeToAge, &selected[c][..9]),
            tail: Tail::from(tails[c]),
        };
        let ex = DevelopmentExhibit::with_averages(&tri, column, &selection, &friedland_averages())
            .unwrap();
        assert_eq!(ex.ages, AGES);
        assert_eq!(ex.link_labels()[0], "12-24");
        assert_eq!(ex.link_labels()[9], "120-ult");
        for (k, (name, factors)) in ex.averages.iter().enumerate() {
            let what = format!("{column} {name}");
            if what == "paid medial latest 5x1" {
                // The book prints 1.010 at 72-84, where the medial average of
                // the printed claims is 1.0094965, 3.5e-6 under the rounding
                // edge: the spreadsheet holds the claims with decimals.
                assert!((factors[5] - 1.0095).abs() < 1e-5, "{what}: {}", factors[5]);
                let mut f = factors.clone();
                f[5] = 1.010;
                assert_printed(&what, &f, &book[k], 3);
            } else {
                assert_printed(&what, factors, &book[k], 3);
            }
        }
        let first: Vec<f64> = ex.link_ratios[..9].iter().map(|r| r[0]).collect();
        assert_printed(&format!("{column} 12-24"), &first, &book_first_link[c], 3);
        assert_printed(
            &format!("{column} 1998"),
            &ex.link_ratios[0],
            &book_1998[c],
            3,
        );
        assert_eq!(ex.selected, selected[c]);
        assert_eq!(ex.tail, tails[c]);
        assert_printed(&format!("{column} CDF"), &ex.cdf, &book_cdf[c], 3);

        // The book divides by the rounded CDFs for the percents; given as a
        // pattern of factors to ultimate, they are the exhibit's own.
        let rounded = ChainLadder {
            development: pattern(PatternKind::AgeToUltimate, &book_cdf[c]),
            ..Default::default()
        };
        let ex = DevelopmentExhibit::with_averages(&tri, column, &rounded, &[]).unwrap();
        let percent: Vec<f64> = ex.percent_developed.iter().map(|p| 100.0 * p).collect();
        assert_printed(&format!("{column} percent"), &percent, &book_percent[c], 1);
        assert_eq!(ex.tail, tails[c]);
        assert!(ex.averages.is_empty());
    }
    // `new` is the Friedland averages with the given development, no tail.
    let plain = DevelopmentExhibit::new(&tri, "reported", &Development::default()).unwrap();
    assert_eq!(plain.averages.len(), 6);
    assert_eq!(plain.tail, 1.0);
}

/// Chapter 7, Exhibit I, sheets 3 and 4: the development method's
/// ultimates, case outstanding, IBNR and unpaid; with chapters 8 to 10's
/// expected claims, Bornhuetter-Ferguson and Cape Cod totals.
#[test]
fn friedland_ultimate_exhibit() {
    let tri = us();
    let cdf = |column: &str| -> ChainLadder {
        let values: &[f64] = if column == "reported" {
            &[
                1.292, 1.110, 1.051, 1.023, 1.011, 1.006, 1.003, 1.001, 1.000, 1.000,
            ]
        } else {
            &[
                2.390, 1.404, 1.184, 1.085, 1.040, 1.020, 1.011, 1.006, 1.004, 1.002,
            ]
        };
        ChainLadder {
            development: pattern(PatternKind::AgeToUltimate, values),
            ..Default::default()
        }
    };
    let apriori = Apriori::by_origin(
        US_YEARS
            .zip([0.75, 0.75, 0.75, 0.75, 0.75, 0.65, 0.65, 0.65, 0.65, 0.65])
            .map(|(y, r)| (Period::year(y), r)),
    );
    let dev_r = cdf("reported").fit(&tri, "reported").unwrap();
    let dev_p = cdf("paid").fit(&tri, "paid").unwrap();
    let ec = ExpectedLoss {
        apriori: apriori.clone(),
        chain_ladder: cdf("reported"),
    }
    .fit(&tri, "reported", "premium")
    .unwrap();
    let bf_r = BornhuetterFerguson {
        apriori: apriori.clone(),
        chain_ladder: cdf("reported"),
    }
    .fit(&tri, "reported", "premium")
    .unwrap();
    let bf_p = BornhuetterFerguson {
        apriori,
        chain_ladder: cdf("paid"),
    }
    .fit(&tri, "paid", "premium")
    .unwrap();
    let cc = CapeCod {
        chain_ladder: cdf("reported"),
        ..Default::default()
    }
    .fit(&tri, "reported", "premium")
    .unwrap();
    let ex = UltimateExhibit::from_triangle(&tri, Some("reported"), Some("paid"))
        .unwrap()
        .with_fit("development reported", &dev_r)
        .unwrap()
        .with_fit("development paid", &dev_p)
        .unwrap()
        .with_fit("expected claims", &ec)
        .unwrap()
        .with_fit("bf reported", &bf_r)
        .unwrap()
        .with_fit("bf paid", &bf_p)
        .unwrap()
        .with_fit("cape cod", &cc)
        .unwrap();
    assert_eq!(ex.ages, AGES.iter().rev().copied().collect::<Vec<_>>());
    // Sheet 4, columns (4) to (10).
    let book_ult_r = [
        47742304.0, 51185767.0, 54892767.0, 56468461.0, 58944268.0, 58198563.0, 58287120.0,
        59682517.0, 60651886.0, 63118803.0,
    ];
    let book_ult_p = [
        47739475.0, 51204536.0, 54860424.0, 56493084.0, 58963359.0, 58167880.0, 58345519.0,
        59963673.0, 61223522.0, 65079626.0,
    ];
    let book_case = [
        98117.0, 185233.0, 304704.0, 421141.0, 785497.0, 1634690.0, 3201985.0, 6141416.0,
        11034842.0, 21623594.0,
    ];
    let book_ibnr_r = [
        0.0, 0.0, 54838.0, 168899.0, 351556.0, 633219.0, 1310463.0, 2896107.0, 6010547.0,
        14265240.0,
    ];
    let book_ibnr_p = [
        -2829.0, 18769.0, 22495.0, 193522.0, 370647.0, 602536.0, 1368862.0, 3177263.0, 6582183.0,
        16226063.0,
    ];
    let book_unpaid_r = [
        98117.0, 185233.0, 359542.0, 590040.0, 1137053.0, 2267909.0, 4512448.0, 9037523.0,
        17045389.0, 35888834.0,
    ];
    let book_unpaid_p = [
        95288.0, 204002.0, 327199.0, 614663.0, 1156144.0, 2237226.0, 4570847.0, 9318679.0,
        17617025.0, 37849657.0,
    ];
    let round = |v: Vec<f64>| -> Vec<f64> { v.iter().map(|x| x.round()).collect() };
    let ult = |m: &str| round(ex.ultimate(m).unwrap().to_vec());
    assert_eq!(ult("development reported"), book_ult_r);
    assert_eq!(ult("development paid"), book_ult_p);
    assert_eq!(round(ex.case_outstanding().unwrap()), book_case);
    assert_eq!(round(ex.ibnr("development reported").unwrap()), book_ibnr_r);
    assert_eq!(round(ex.ibnr("development paid").unwrap()), book_ibnr_p);
    assert_eq!(
        round(ex.unpaid("development reported").unwrap()),
        book_unpaid_r
    );
    assert_eq!(round(ex.unpaid("development paid").unwrap()), book_unpaid_p);
    // Totals of sheet 4 and of chapters 8 (Exhibit II, sheet 2), 9
    // (Exhibit I, sheet 1) and 10 (Exhibit I): ultimate, IBNR, total unpaid.
    let book_totals = [
        (
            "development reported",
            [569172456.0, 25690869.0, 71122088.0],
        ),
        ("development paid", [572041099.0, 28559512.0, 73990731.0]),
        ("expected claims", [569281839.0, 25800252.0, 71231471.0]),
        ("bf reported", [569091348.0, 25609761.0, 71040980.0]),
        ("bf paid", [570568198.0, 27086611.0, 72517830.0]),
        ("cape cod", [570800677.0, 27319090.0, 72750309.0]),
    ];
    let sum = |v: &[f64]| -> f64 { v.iter().sum::<f64>().round() };
    for (m, book) in book_totals {
        let got = [
            sum(ex.ultimate(m).unwrap()),
            sum(&ex.ibnr(m).unwrap()),
            sum(&ex.unpaid(m).unwrap()),
        ];
        assert_eq!(got, book, "{m}");
    }
    assert_eq!(sum(&ex.case_outstanding().unwrap()), 45431219.0);
    let table = ex.ultimates_table();
    assert_eq!(table.values.len(), 8);
    assert_eq!(ex.ibnr_table().unwrap().values.len(), 6);
    // A single method's selection is that method.
    let single = ex
        .select(&UltimateSelection::new(MethodWeights::Single(
            "bf paid".into(),
        )))
        .unwrap();
    assert_eq!(single.ultimate, ex.ultimate("bf paid").unwrap());
    assert_eq!(single.total_ibnr().unwrap().round(), 27086611.0);
}

/// Chapter 15, Exhibit I, sheets 2 and 6 (XYZ Insurer, Auto BI): the
/// selected ultimate of column (11), a weighting by accident year of the
/// printed projections of chapters 11 and 13. The book holds the claims with
/// decimals and prints them rounded, so the selection is checked to 1.
#[test]
fn friedland_chapter_15_selection() {
    let origins: Vec<Period> = (1998..=2008).map(Period::year).collect();
    let ages: Vec<u32> = (0..11).map(|k| 132 - 12 * k).collect();
    let nan = f64::NAN;
    // Sheet 2, columns (2) to (10).
    let reported = vec![
        15822.0, 25107.0, 37246.0, 38798.0, 48169.0, 44373.0, 70288.0, 70655.0, 48804.0, 31732.0,
        18632.0,
    ];
    let paid = vec![
        15822.0, 24817.0, 36782.0, 38519.0, 44437.0, 39320.0, 52811.0, 40026.0, 22819.0, 11865.0,
        3409.0,
    ];
    let methods: [(&str, [f64; 11]); 7] = [
        (
            "development both",
            [
                15822.0, 25107.0, 37246.0, 38798.0, 48169.0, 44373.0, 70780.0, 71362.0, 53392.0,
                42680.0, 41531.0,
            ],
        ),
        (
            "development paid",
            [
                15980.0, 25140.0, 37334.0, 39290.0, 47326.0, 44078.0, 71401.0, 68685.0, 51776.0,
                38549.0, 45987.0,
            ],
        ),
        (
            "bf reported",
            [
                15822.0, 25107.0, 37246.0, 38798.0, 48169.0, 44373.0, 70792.0, 71554.0, 53906.0,
                40300.0, 36842.0,
            ],
        ),
        (
            "bf paid",
            [
                15975.0, 25128.0, 37294.0, 39274.0, 47313.0, 45070.0, 71688.0, 77898.0, 56031.0,
                34988.0, 33988.0,
            ],
        ),
        (
            "frequency-severity 1",
            [
                15822.0, 25082.0, 37083.0, 38778.0, 48655.0, 46107.0, 76620.0, 80745.0, 64505.0,
                58516.0, 59242.0,
            ],
        ),
        (
            "frequency-severity 2",
            [
                nan, nan, nan, nan, nan, nan, nan, nan, nan, 30512.0, 30140.0,
            ],
        ),
        (
            "frequency-severity 3",
            [
                nan, nan, nan, 39192.0, 46869.0, 44479.0, 71906.0, 71684.0, 49913.0, 31805.0,
                29828.0,
            ],
        ),
    ];
    let mut ex = UltimateExhibit::new(origins, ages, Some(reported), Some(paid)).unwrap();
    for (name, u) in methods {
        ex = ex.with_method(name, u.to_vec()).unwrap();
    }
    // Note (11): column (4) for 2004 and prior, the average of (6) and (7)
    // for 2005 and 2006, (6) for 2007 and the average of (6) and (9) for 2008.
    let by_origin = MethodWeights::ByOrigin(vec![
        (
            "development both".into(),
            [vec![1.0; 7], vec![0.0; 4]].concat(),
        ),
        (
            "bf reported".into(),
            [vec![0.0; 7], vec![0.5, 0.5, 1.0, 0.5]].concat(),
        ),
        (
            "bf paid".into(),
            [vec![0.0; 7], vec![0.5, 0.5, 0.0, 0.0]].concat(),
        ),
        (
            "frequency-severity 2".into(),
            [vec![0.0; 10], vec![0.5]].concat(),
        ),
    ]);
    // The same rule by age at 12/31/08: 60 months and older, 36 and 48,
    // 24, and 12.
    let by_age = MethodWeights::ByAge(vec![
        (
            12,
            vec![
                ("bf reported".into(), 0.5),
                ("frequency-severity 2".into(), 0.5),
            ],
        ),
        (24, vec![("bf reported".into(), 1.0)]),
        (
            36,
            vec![("bf reported".into(), 0.5), ("bf paid".into(), 0.5)],
        ),
        (60, vec![("development both".into(), 1.0)]),
    ]);
    let book_selected = [
        15822.0, 25107.0, 37246.0, 38798.0, 48169.0, 44373.0, 70780.0, 74726.0, 54968.0, 40300.0,
        33491.0,
    ];
    let book_ibnr = [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 492.0, 4071.0, 6164.0, 8568.0, 14859.0,
    ];
    for weights in [by_origin, by_age] {
        let sel = ex.select(&UltimateSelection::new(weights)).unwrap();
        assert_close("selected", &sel.ultimate, &book_selected, 0.5);
        assert_close("selected IBNR", &sel.ibnr().unwrap(), &book_ibnr, 0.5);
        assert!((sel.total_ultimate() - 483781.0).abs() <= 1.0);
        assert!((sel.total_ibnr().unwrap() - 34155.0).abs() <= 1.0);
        assert!((sel.total_unpaid().unwrap() - (483781.0 - 330629.0)).abs() <= 3.0);
    }
    // Sheet 2's totals of the methods projecting every year.
    for (m, book) in [
        ("development both", 489258.0),
        ("development paid", 485546.0),
        ("bf reported", 482910.0),
        ("bf paid", 484648.0),
        ("frequency-severity 1", 551155.0),
    ] {
        let total: f64 = ex.ultimate(m).unwrap().iter().sum();
        assert!((total - book).abs() <= 2.0, "{m}: {total} against {book}");
    }
    // Weights that do not sum to 1 in 2008 are refused, not rescaled.
    let short = MethodWeights::ByOrigin(vec![
        (
            "development both".into(),
            [vec![1.0; 10], vec![0.5]].concat(),
        ),
        ("bf reported".into(), [vec![0.0; 10], vec![0.25]].concat()),
    ]);
    let err = ex.select(&UltimateSelection::new(short)).unwrap_err();
    assert!(err.to_string().contains("2008 sum to 0.75"), "{err}");
}

/// Every row of `reserving_exhibits_python.csv`: VotingChainladder's
/// ultimates and reserves on GenIns, through `UltimateExhibit::select`.
#[test]
fn voting_chainladder_parity() {
    let tri = triangle_columns("genins_premium", &["paid", "premium"]);
    let cl = ChainLadder::default().fit(&tri, "paid").unwrap();
    let bf = BornhuetterFerguson {
        apriori: 0.75.into(),
        ..Default::default()
    }
    .fit(&tri, "paid", "premium")
    .unwrap();
    let cc = CapeCod::default().fit(&tri, "paid", "premium").unwrap();
    let ex = UltimateExhibit::from_triangle(&tri, None, Some("paid"))
        .unwrap()
        .with_fit("cl", &cl)
        .unwrap()
        .with_fit("bf", &bf)
        .unwrap()
        .with_fit("cc", &cc)
        .unwrap();
    let rows = |w: [[f64; 3]; 10]| -> MethodWeights {
        MethodWeights::ByOrigin(
            ["cl", "bf", "cc"]
                .iter()
                .enumerate()
                .map(|(m, name)| (name.to_string(), w.iter().map(|r| r[m]).collect()))
                .collect(),
        )
    };
    let phased = {
        let mut w = [[0.0; 3]; 10];
        for (i, r) in w.iter_mut().enumerate() {
            *r = match i {
                0..=3 => [1.0, 0.0, 0.0],
                4..=6 => [0.5, 0.5, 0.0],
                7..=8 => [0.0, 0.5, 0.5],
                _ => [0.0, 0.0, 1.0],
            };
        }
        w
    };
    let cases = [
        ("phased", rows(phased)),
        ("blend", rows([[0.5, 0.25, 0.25]; 10])),
        ("single_bf", MethodWeights::Single("bf".into())),
    ];
    let selected: Vec<_> = cases
        .into_iter()
        .map(|(name, w)| (name, ex.select(&UltimateSelection::new(w)).unwrap()))
        .collect();
    let origin = |case: &Case| -> usize {
        let year: i32 = case.get("arg").parse().unwrap();
        ex.origins
            .iter()
            .position(|p| *p == Period::year(year))
            .unwrap()
    };
    let cases = reference("reserving_exhibits_python.csv");
    check(&cases, |case| {
        let (_, sel) = selected.iter().find(|(n, _)| *n == case.get("method"))?;
        let quantity = case.get("quantity");
        if let Some(method) = quantity.strip_prefix("ultimate:") {
            return Some(ex.ultimate(method)?[origin(case)]);
        }
        Some(match quantity {
            "ultimate" => sel.ultimate[origin(case)],
            "reserve" => sel.unpaid()?[origin(case)],
            "total_ultimate" => sel.total_ultimate(),
            "total_reserve" => sel.total_unpaid()?,
            _ => return None,
        })
    });
}
