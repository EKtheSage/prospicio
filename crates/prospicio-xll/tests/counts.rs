//! The methods on claim counts (`docs/design/xll.md`, decision 52) against
//! Friedland, *Estimating Unpaid Claims Using Basic Techniques* (2010),
//! chapters 11 to 13, as `validation/tests/reserving_counts.rs` checks the
//! library, and against the library itself where the book's data does not
//! fit a sheet triangle. A sheet triangle is observed from its first
//! column, so XYZ Insurer is used from accident year 2001 (its counts start
//! in calendar year 2001).

mod common;

use common::{cell, data_window, fails, n, ok, row, t};
use prospicio_reserving::{
    BerquistSherman, CaseAdequacy, CaseTrend, DevelopmentColumn, Grain, Interpolation, Long, Month,
    OpenCounts, Selection, SettlementRate, Triangle, UltimateCounts,
};
use prospicio_xll::sheet::{Grid, Value};

fn close(got: f64, want: f64, tol: f64, what: &str) {
    assert!(
        (got - want).abs() <= tol,
        "{what}: got {got}, want {want} (tolerance {tol})"
    );
}

/// XYZ Insurer, accident years 2001 to 2008 to 96 months.
fn xyz(column: &str) -> (Grid, Vec<f64>) {
    data_window("friedland_xyz", column, 2001, 2008, 96)
}

fn years(first: i32, last: i32) -> Grid {
    Grid::column(&(first..=last).map(f64::from).collect::<Vec<_>>())
}

fn column(table: &Grid, name: &str) -> Vec<f64> {
    let c = common::column_of(table, name);
    (1..table.rows - 1)
        .map(|r| cell(table, r, c).unwrap())
        .collect()
}

fn total(table: &Grid, name: &str) -> f64 {
    cell(table, table.rows - 1, common::column_of(table, name)).unwrap()
}

/// Chapter 11, Exhibit II: approach #1. The ultimate counts are the mean
/// of the printed closed and reported count projections, the severities
/// developed with the printed factors to ultimate (`kind` "cdf", the last
/// one the tail).
#[test]
fn friedland_11_approach_1() {
    let (reported, _) = xyz("reported");
    let (counts, _) = xyz("reported_counts");
    let closed = [1476., 1576., 1701., 2356., 2565., 1936., 1804., 1679.];
    let reported_ult = [1455., 1554., 1631., 2263., 2402., 1679., 1308., 1172.];
    let ultimate: Vec<f64> = closed
        .iter()
        .zip(reported_ult)
        .map(|(c, r)| (c + r) / 2.0)
        .collect();
    let cdf = [2.310, 1.498, 1.212, 1.100, 1.066, 1.016, 1.003, 0.992];
    let table = ok(
        "pro_res_freqsev",
        vec![
            reported,
            counts,
            Grid::column(&ultimate),
            Grid::missing(),
            Grid::missing(),
            row(&cdf),
            t("cdf"),
            years(2001, 2008),
        ],
    );
    assert_eq!(*table.get(1, 0), Value::Number(2001.0));
    let latest = [
        26665., 30997., 27239., 31129., 29563., 29453., 25104., 17985.,
    ];
    for (k, s) in column(&table, "Latest severity").iter().enumerate() {
        close(s * 1000.0, latest[k], 1.5, "latest severity");
    }
    let sev = [
        26452., 31090., 27675., 33183., 32519., 35697., 37606., 41544.,
    ];
    for (k, s) in column(&table, "Ultimate severity").iter().enumerate() {
        close(s * 1000.0, sev[k], 2.0, "ultimate severity");
    }
    // The printed counts are rounded: within 0.08%.
    let want = [
        38778., 48655., 46107., 76620., 80745., 64505., 58516., 59242.,
    ];
    for (k, u) in column(&table, "Ultimate").iter().enumerate() {
        close(*u, want[k], 8e-4 * want[k], "ultimate");
    }
    let ult: f64 = column(&table, "Ultimate").iter().sum();
    close(total(&table, "Ultimate"), ult, 1e-6, "total");
    close(
        total(&table, "Reserve"),
        ult - column(&table, "Latest").iter().sum::<f64>(),
        1e-6,
        "reserve",
    );
}

/// Chapter 11, Exhibit IV: approach #2, a 2.36% frequency on premium at
/// 2008 rates (2007's on-level factor 0.800) with a -1.5% count trend, and
/// a $26,720 severity with a 5% trend, both at the 2008 level.
#[test]
fn friedland_11_approach_2() {
    let (reported, _) = xyz("reported");
    let (counts, _) = xyz("reported_counts");
    let (_, premium) = xyz("premium");
    let mut on_level = vec![1.0; 8];
    on_level[6] = 0.8;
    let table = ok(
        "pro_res_freqsev_trended",
        vec![
            reported,
            counts,
            Grid::column(&premium),
            n(0.0236),
            n(26.720),
            n(2008.0),
            n(-0.015),
            n(0.05),
            Grid::column(&on_level),
            Grid::missing(),
            Grid::missing(),
            years(2001, 2008),
        ],
    );
    let counts = column(&table, "Ultimate counts");
    close(
        counts[6],
        62438.0 * 0.0236 * 0.8 / 0.985,
        1e-9,
        "2007 counts",
    );
    close(counts[7], 1128.0, 0.5, "2008 counts");
    close(
        column(&table, "Ultimate severity")[6] * 1000.0,
        25448.0,
        0.5,
        "2007 severity",
    );
    let ult = column(&table, "Ultimate");
    close(ult[6], 30512.152, 0.0025 * 30512.152, "2007 ultimate");
    close(ult[7], 30140.260, 0.5, "2008 ultimate");
    close(column(&table, "Reserve")[7], 11508.260, 0.5, "2008 IBNR");
    // The base defaults to the latest origin.
    let mut args = vec![
        xyz("reported").0,
        xyz("reported_counts").0,
        Grid::column(&premium),
        n(0.0236),
        n(26.720),
        Grid::missing(),
        n(-0.015),
        n(0.05),
        Grid::column(&on_level),
        Grid::missing(),
        Grid::missing(),
        Grid::missing(),
    ];
    let unlabelled = ok("pro_res_freqsev_trended", args.clone());
    close(
        column(&unlabelled, "Ultimate")[7],
        ult[7],
        1e-9,
        "default base",
    );
    args[5] = n(1999.0);
    assert!(fails("pro_res_freqsev_trended", args).contains("base"));
}

/// Chapter 11, Exhibit VI: the disposal rate method on XYZ Insurer, 2001
/// to 2008, with the latest-2 averages, a 5% trend to 2008 and the tort
/// reform factors, the pooled severity of 84 months and older beyond.
#[test]
fn friedland_11_disposal() {
    let (closed, _) = xyz("closed_counts");
    let (paid, _) = xyz("paid");
    let reported = [1455., 1554., 1629., 2258., 2390., 1657., 1264., 1036.];
    let cdf = [1.000, 1.000, 1.001, 1.002, 1.005, 1.013, 1.035, 1.131];
    let ultimate: Vec<f64> = reported.iter().zip(cdf).map(|(r, c)| r * c).collect();
    let mut factors = vec![0.670; 5];
    factors.extend([0.750, 1.0, 1.0]);
    let args = |rates: Grid, output: &str| {
        vec![
            closed.clone(),
            paid.clone(),
            Grid::column(&ultimate),
            rates,
            t("latest 2"),
            n(84.0),
            n(0.05),
            n(2008.0),
            Grid::column(&factors),
            years(2001, 2008),
            t(output),
        ]
    };
    // Sheet 1, part 3: the selected rates.
    let printed = [0.244, 0.572, 0.704, 0.814, 0.910, 0.952, 0.982, 0.994];
    let rates = ok("pro_res_disposal", args(n(2.0), "rates"));
    let got: Vec<f64> = (0..8)
        .map(|c| (cell(&rates, 0, c).unwrap() * 1000.0).round() / 1000.0)
        .collect();
    assert_eq!(got, printed);
    // The book projects with the rates rounded; given as a row.
    let counts = ok("pro_res_disposal", args(row(&printed), "counts"));
    let want = [276., 389., 156., 130., 114., 50., 36., 14., 7.];
    for (c, w) in want.iter().enumerate() {
        close(cell(&counts, 7, c).unwrap(), *w, 0.5, "2008 counts");
    }
    let sev = ok("pro_res_disposal", args(row(&printed), "severities"));
    let want = [
        11807., 15165., 26043., 35183., 41908., 62206., 70432., 70432., 70432.,
    ];
    for (c, w) in want.iter().enumerate() {
        close(cell(&sev, 0, c).unwrap() * 1000.0, *w, 10.0, "severity");
    }
    let table = ok("pro_res_disposal", args(row(&printed), "ultimate"));
    let want = [
        39192., 46869., 44479., 71906., 71684., 49913., 31805., 29828.,
    ];
    for (k, u) in column(&table, "Ultimate").iter().enumerate() {
        close(*u, want[k], 2.0, "ultimate");
    }
    close(total(&table, "Ultimate"), 385676.0, 6.0, "total ultimate");
    close(total(&table, "Unpaid"), 385676.0 - 253208.0, 6.0, "unpaid");
    // The payments square holds the latest paid and the projection.
    let payments = ok("pro_res_disposal", args(row(&printed), "payments"));
    let projected: f64 = (0..payments.columns)
        .filter_map(|c| cell(&payments, 7, c))
        .sum();
    close(
        projected,
        column(&table, "Ultimate")[7],
        1e-6,
        "2008 payments",
    );
    assert!(fails("pro_res_disposal", args(t("latest"), "ultimate")).contains("rates"));
    assert!(fails("pro_res_disposal", args(n(2.0), "everything")).contains("output"));
}

/// Chapter 12, Exhibit I: U.S. Industry Auto, the ratios selected by the
/// simple average of the latest 3, 1.10 paid to ultimate.
#[test]
fn friedland_12_case_outstanding() {
    let (paid, _) = data_window("friedland_us_auto", "paid", 1998, 2007, 120);
    let (reported, _) = data_window("friedland_us_auto", "reported", 1998, 2007, 120);
    let args = |poc: Grid, coc: Grid, output: &str| {
        vec![
            paid.clone(),
            reported.clone(),
            poc,
            coc,
            n(1.10),
            years(1998, 2007),
            t(output),
        ]
    };
    let paid_on_case = [
        0.833, 0.701, 0.714, 0.714, 0.653, 0.631, 0.553, 0.437, 0.524,
    ];
    let case_on_case = [
        0.526, 0.566, 0.528, 0.486, 0.511, 0.555, 0.632, 0.674, 0.580,
    ];
    for (output, want) in [
        ("paid_on_case", paid_on_case),
        ("case_on_case", case_on_case),
    ] {
        let r = ok("pro_res_case_os", args(t("latest 3"), n(3.0), output));
        let got: Vec<f64> = (1..10)
            .map(|c| (cell(&r, 0, c).unwrap() * 1000.0).round() / 1000.0)
            .collect();
        assert_eq!(got, want, "{output}");
    }
    // The book projects with the rounded ratios: a row under the triangle's
    // columns, blank at 12 months.
    let given = |v: &[f64]| {
        let mut cells = vec![Value::Empty];
        cells.extend(v.iter().map(|&x| Value::Number(x)));
        Grid::from_rows(vec![cells])
    };
    let table = ok(
        "pro_res_case_os",
        args(given(&paid_on_case), given(&case_on_case), "ultimate"),
    );
    let want = [
        47752116., 51215775., 54905021., 56436079., 58880132., 58139086., 58163235., 59474287.,
        60321169., 62470839.,
    ];
    for (k, u) in column(&table, "Ultimate").iter().enumerate() {
        close(*u, want[k], 1.0, "ultimate");
    }
    close(
        total(&table, "Ultimate"),
        567757738.0,
        2.0,
        "total ultimate",
    );
    close(total(&table, "Unpaid"), 69707370.0, 2.0, "unpaid");
    close(total(&table, "IBNR"), 24276151.0, 2.0, "IBNR");
    let case = ok(
        "pro_res_case_os",
        args(given(&paid_on_case), given(&case_on_case), "case"),
    );
    close(
        cell(&case, 9, 1).unwrap(),
        11374010.0,
        1.0,
        "2007 case at 24",
    );
}

/// Berquist and Sherman's medical malpractice data, chapter 13, Exhibit I.
fn medmal(column: &str) -> Grid {
    common::data_triangle("berqsherm_medmal", column).0
}

fn numbers(g: &Grid) -> Vec<Vec<Option<f64>>> {
    (0..g.rows)
        .map(|r| (0..g.columns).map(|c| cell(g, r, c)).collect())
        .collect()
}

fn minus(a: &Grid, b: &Grid) -> Grid {
    let (a, b) = (numbers(a), numbers(b));
    Grid::from_rows(
        a.iter()
            .zip(&b)
            .map(|(x, y)| {
                x.iter()
                    .zip(y)
                    .map(|(x, y)| match (x, y) {
                        (Some(x), Some(y)) => Value::Number(x - y),
                        _ => Value::Empty,
                    })
                    .collect()
            })
            .collect(),
    )
}

/// Chapter 13, Exhibit I: the trend of the average case outstanding by
/// age (sheet 4), and the case adequacy adjustment at 15% (sheet 6).
#[test]
fn friedland_13_case_adequacy() {
    let open = minus(&medmal("reported"), &medmal("closed"));
    let case = |trend: Grid, output: &str| {
        ok(
            "pro_res_bs_case",
            vec![
                medmal("incurred"),
                medmal("paid"),
                open.clone(),
                trend,
                Grid::missing(),
                t(output),
            ],
        )
    };
    let average = case(n(0.15), "average_case");
    let trends = ok("pro_res_trend_by_age", vec![average, Grid::missing()]);
    assert_eq!(*trends.get(1, 0), Value::Number(12.0));
    let rate: Vec<f64> = (1..8)
        .map(|r| (cell(&trends, r, 1).unwrap() * 1000.0).round() / 10.0)
        .collect();
    assert_eq!(rate, [15.6, 29.5, 31.1, 34.2, 33.0, 32.2, 27.6]);
    let r2: Vec<f64> = (1..8)
        .map(|r| (cell(&trends, r, 2).unwrap() * 1000.0).round() / 10.0)
        .collect();
    assert_eq!(r2, [80.0, 89.5, 85.8, 94.1, 98.9, 98.3, 100.0]);
    assert_eq!(*trends.get(trends.rows - 1, 0), Value::from("All ages"));
    // The same trends from the case outstanding over the open counts.
    let outstanding = minus(&medmal("incurred"), &medmal("paid"));
    let again = ok("pro_res_trend_by_age", vec![outstanding, open.clone()]);
    for r in 1..trends.rows {
        for c in 1..4 {
            match (cell(&trends, r, c), cell(&again, r, c)) {
                (Some(a), Some(b)) => close(a, b, 1e-12, "trend"),
                (a, b) => assert_eq!(a, b),
            }
        }
    }
    let adjusted = numbers(&case(n(0.15), "adjusted"));
    let want: [&[f64]; 8] = [
        &[
            3793504., 12084942., 18563821., 25924316., 23516364., 24979245., 24016864., 23506000.,
        ],
        &[
            3760482., 15830500., 24615996., 33169802., 30722141., 33362729., 32216000.,
        ],
        &[
            5982185., 25583831., 41384825., 50323342., 46191356., 48377000.,
        ],
        &[7819355., 33794110., 51361061., 64559286., 61163000.],
        &[9533246., 34585431., 49667342., 73733000.],
        &[10348458., 41241243., 63477000.],
        &[13102479., 48904000.],
        &[15791000.],
    ];
    for (o, r) in want.iter().enumerate() {
        for (d, w) in r.iter().enumerate() {
            close(adjusted[o][d].unwrap(), *w, 0.5, "adjusted incurred");
        }
        assert!(adjusted[o][r.len()..].iter().all(Option::is_none));
    }
    assert!(
        fails(
            "pro_res_bs_case",
            vec![medmal("incurred"), medmal("paid"), open, Grid::missing()]
        )
        .contains("trend")
    );
}

/// Berquist and Sherman's automobile bodily injury data, chapter 13,
/// Exhibit II: the settlement rate adjustment on the latest reported counts
/// times the printed factors to ultimate.
#[test]
fn friedland_13_settlement_rate() {
    let auto = |c: &str| common::data_triangle("berqsherm_auto", c).0;
    let reported = numbers(&auto("reported"));
    let cdf = [1.000, 1.000, 1.000, 1.001, 1.003, 1.008, 1.021, 1.221];
    let ultimate: Vec<f64> = reported
        .iter()
        .zip(cdf)
        .map(|(r, c)| r.iter().flatten().last().unwrap() * c)
        .collect();
    let settle = |output: &str| {
        numbers(&ok(
            "pro_res_bs_settlement",
            vec![
                auto("paid"),
                auto("closed"),
                Grid::column(&ultimate),
                Grid::missing(),
                Grid::missing(),
                t(output),
            ],
        ))
    };
    let closed: [&[f64]; 8] = [
        &[3383., 6049., 6932., 7415., 7643., 7750., 7789., 7806.],
        &[3756., 6715., 7695., 8231., 8484., 8603., 8647.],
        &[4302., 7692., 8815., 9429., 9719., 9855.],
        &[4192., 7495., 8588., 9187., 9469.],
        &[4149., 7418., 8501., 9093.],
        &[3376., 6035., 6916.],
        &[3482., 6226.],
        &[3230.],
    ];
    let paid: [&[f64]; 8] = [
        &[1431., 4277., 6463., 8497., 9579., 10075., 10191., 10256.],
        &[1745., 5181., 7865., 10156., 11301., 11744., 12031.],
        &[1985., 6243., 9594., 12233., 13550., 14235.],
        &[2247., 7228., 11072., 13837., 15383.],
        &[2585., 7999., 11982., 15278.],
        &[2292., 7271., 11771.],
        &[2718., 9182.],
        &[2801.],
    ];
    let (got_closed, got_paid) = (settle("closed"), settle("paid"));
    for o in 0..8 {
        for d in 0..closed[o].len() {
            close(got_closed[o][d].unwrap(), closed[o][d], 0.5, "closed");
            close(got_paid[o][d].unwrap(), paid[o][d], 0.5, "paid");
        }
    }
    // Linear interpolation moves the earlier cells but not the latest.
    let linear = numbers(&ok(
        "pro_res_bs_settlement",
        vec![
            auto("paid"),
            auto("closed"),
            Grid::column(&ultimate),
            t("diagonal"),
            t("linear"),
        ],
    ));
    close(
        linear[0][7].unwrap(),
        got_paid[0][7].unwrap(),
        1e-9,
        "latest",
    );
    assert!((linear[0][0].unwrap() - got_paid[0][0].unwrap()).abs() > 1.0);
}

/// The medical malpractice triangle with every column, as the library
/// reads it.
fn medmal_triangle() -> Triangle {
    let names = ["incurred", "paid", "reported", "closed"];
    let grids: Vec<Vec<Vec<Option<f64>>>> = names.iter().map(|c| numbers(&medmal(c))).collect();
    let (mut origin, mut age) = (Vec::new(), Vec::new());
    let mut values: Vec<Vec<f64>> = vec![Vec::new(); names.len()];
    for (o, r) in grids[0].iter().enumerate() {
        for (d, v) in r.iter().enumerate() {
            if v.is_some() {
                origin.push(Month::january(1969 + o as i32));
                age.push(12 * (d as u32 + 1));
                for (k, g) in grids.iter().enumerate() {
                    values[k].push(g[o][d].unwrap());
                }
            }
        }
    }
    let columns: Vec<(&str, &[f64])> = names
        .iter()
        .zip(&values)
        .map(|(n, v)| (*n, v.as_slice()))
        .collect();
    Triangle::from_long(&Long {
        keys: &[],
        origin: &origin,
        development: DevelopmentColumn::Age(&age),
        values: &columns,
        origin_grain: Grain::Year,
        development_grain: Grain::Year,
        cumulative: true,
    })
    .unwrap()
}

/// Both adjustments agree with the library's `BerquistSherman`, cell by
/// cell, on the medical malpractice data at a 15% trend to 60 months.
#[test]
fn both_adjustments_match_the_library() {
    let tri = medmal_triangle();
    let ultimate = vec![4100., 4600., 5100., 5900., 6400., 6600., 6900., 7100.];
    let both = BerquistSherman {
        case_adequacy: CaseAdequacy {
            incurred: "incurred".into(),
            paid: "paid".into(),
            open_counts: OpenCounts::ReportedLessClosed {
                reported: "reported".into(),
                closed: "closed".into(),
            },
            trend: CaseTrend::Given(0.15),
            max_age: Some(60),
        },
        settlement_rate: SettlementRate {
            paid: "paid".into(),
            closed_counts: "closed".into(),
            ultimate_counts: UltimateCounts::Given(ultimate.clone()),
            disposal_rates: Selection::LatestDiagonal,
            interpolation: Interpolation::Exponential,
        },
    }
    .adjust(&tri)
    .unwrap();
    for (output, c) in [("incurred", 0), ("paid", 1), ("closed", 3)] {
        let got = numbers(&ok(
            "pro_res_bs",
            vec![
                medmal("incurred"),
                medmal("paid"),
                medmal("reported"),
                medmal("closed"),
                Grid::column(&ultimate),
                n(0.15),
                n(60.0),
                Grid::missing(),
                Grid::missing(),
                t(output),
            ],
        ));
        for (o, r) in got.iter().enumerate() {
            for (d, v) in r.iter().enumerate() {
                assert_eq!(*v, both.get(0, c, o, d), "{output} {o} {d}");
            }
        }
    }
}

#[test]
fn selections_and_shapes_are_checked() {
    let (closed, _) = xyz("closed_counts");
    let (paid, _) = xyz("paid");
    let ultimate = Grid::column(&[1500.0; 8]);
    let call = |rates: Grid, paid: Grid| {
        prospicio_xll::call(
            "pro_res_disposal",
            vec![closed.clone(), paid, ultimate.clone(), rates],
        )
    };
    let message = |g: Grid| match g.as_scalar() {
        Some(Value::Text(s)) => s.clone(),
        other => panic!("expected an error, got {other:?}"),
    };
    assert!(message(call(t("median 3"), paid.clone())).contains("rates must be"));
    assert!(message(call(n(0.5), paid.clone())).contains("whole number"));
    let short = data_window("friedland_xyz", "paid", 2001, 2007, 96).0;
    assert!(message(call(Grid::missing(), short)).contains("same size"));
    // "all", "medial 5" and "medial 5x1" are averages; medial drops one
    // each side.
    for spec in ["all", "medial 5", "medial 5x1", "latest 3", "diagonal"] {
        ok(
            "pro_res_disposal",
            vec![closed.clone(), paid.clone(), ultimate.clone(), t(spec)],
        );
    }
}
