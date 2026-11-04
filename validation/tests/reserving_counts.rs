//! Parity of the methods on claim counts with Friedland, *Estimating Unpaid
//! Claims Using Basic Techniques* (CAS, version 3, 2010), chapters 11 to 13,
//! and with chainladder-python 0.10.1's `BerquistSherman`.
//!
//! The book's exhibits come from a spreadsheet that holds claims and
//! counts with decimals and prints them rounded, multiplies by factors to
//! ultimate rounded to three decimals and selects some values by judgment.
//! The tests therefore feed the printed selections (factors to ultimate,
//! a severity of 26,720) where the book selects by judgment, check the
//! averages it selects from separately, and compare with the printed digits
//! within the rounding of the data: each tolerance says how much.
//!
//! - Chapter 11: frequency-severity on XYZ Insurer (Exhibits II and IV)
//!   and the disposal rate method (Exhibit VI).
//! - Chapter 12: case outstanding development on U.S. Industry Auto
//!   (Exhibit I) and XYZ Insurer (Exhibit II), and the case development
//!   factor of approach #2 (Exhibit III).
//! - Chapter 13: the Berquist-Sherman adjustments on Berquist and
//!   Sherman's medical malpractice (Exhibit I) and automobile bodily injury
//!   (Exhibit II) examples and on XYZ Insurer (Exhibit III).
//! - chainladder-python's `BerquistSherman` on its `berqsherm` sample
//!   (`reference/reserving_berquist_sherman_python.csv`, written by
//!   `scripts/reserving_berquist_sherman_python.py`).

use std::collections::BTreeMap;

use prospicio_reserving::{
    Average, BerquistSherman, CaseAdequacy, CaseOutstanding, CaseTrend, ChainLadder,
    CountProjection, Development, DevelopmentColumn, Disposal, FrequencySeverity, Grain,
    Interpolation, Lag, Long, Month, OpenCounts, PatternKind, Period, SelectedPattern, Selection,
    SettlementRate, SeverityProjection, Trended, Triangle, UltimateCounts, case_development_factor,
    pooled_trend, ratio_column, trend_by_age,
};
use prospicio_validation::{check, reference};

// ---------------------------------------------------------------------------
// Data
// ---------------------------------------------------------------------------

/// The rows of `validation/data/<name>.csv` as column name to value.
fn rows(name: &str) -> Vec<BTreeMap<String, f64>> {
    let path = format!("{}/data/{name}.csv", env!("CARGO_MANIFEST_DIR"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let mut lines = text
        .lines()
        .filter(|l| !l.starts_with('#') && !l.is_empty());
    let header: Vec<&str> = lines.next().unwrap().split(',').collect();
    lines
        .map(|l| {
            header
                .iter()
                .zip(l.split(','))
                .map(|(h, v)| (h.to_string(), v.parse().unwrap()))
                .collect()
        })
        .collect()
}

/// A cumulative annual triangle of `columns` from `data/<name>.csv`, with
/// only the origins from `first` to `last` and the ages up to `max_age`.
fn load(name: &str, columns: &[&str], first: i32, last: i32, max_age: Lag) -> Triangle {
    let rows: Vec<_> = rows(name)
        .into_iter()
        .filter(|r| {
            let (o, a) = (r["origin"] as i32, r["development"] as Lag);
            (first..=last).contains(&o) && a <= max_age
        })
        .collect();
    let origin: Vec<Month> = rows
        .iter()
        .map(|r| Month::january(r["origin"] as i32))
        .collect();
    let ages: Vec<Lag> = rows.iter().map(|r| r["development"] as Lag).collect();
    let values: Vec<(&str, Vec<f64>)> = columns
        .iter()
        .map(|&c| (c, rows.iter().map(|r| r[c]).collect()))
        .collect();
    let values: Vec<(&str, &[f64])> = values.iter().map(|(n, v)| (*n, v.as_slice())).collect();
    Triangle::from_long(&Long {
        keys: &[],
        origin: &origin,
        development: DevelopmentColumn::Age(&ages),
        values: &values,
        origin_grain: Grain::Year,
        development_grain: Grain::Year,
        cumulative: true,
    })
    .unwrap()
}

const XYZ: &[&str] = &[
    "reported",
    "paid",
    "reported_counts",
    "closed_counts",
    "premium",
];

fn xyz() -> Triangle {
    load("friedland_xyz", XYZ, 1998, 2008, 132)
}

/// A chain ladder whose factors to ultimate are the printed ones, given as
/// the ratios of consecutive printed factors (some are below 1) at the
/// triangle's ages from `first`.
fn printed_cdf(cdf: &[f64], first: Lag) -> ChainLadder {
    let values = (0..cdf.len() - 1)
        .map(|k| (first + 12 * k as Lag, cdf[k] / cdf[k + 1]))
        .collect();
    ChainLadder {
        development: Development {
            pattern: Some(SelectedPattern::new(PatternKind::AgeToAge, values)),
            ..Development::default()
        },
        tail: cdf[cdf.len() - 1].into(),
    }
}

fn close(got: f64, want: f64, tol: f64, what: &str) {
    assert!(
        (got - want).abs() <= tol,
        "{what}: got {got}, want {want} (tolerance {tol})"
    );
}

fn close_all(got: &[f64], want: &[f64], tol: f64, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: lengths");
    for (k, (g, w)) in got.iter().zip(want).enumerate() {
        close(*g, *w, tol, &format!("{what}[{k}]"));
    }
}

/// The volume-weighted (or simple) average factors over the latest `n`
/// diagonals, rounded to three decimals as the book prints them.
fn averages(tri: &Triangle, column: &str, average: Average, n: usize) -> Vec<f64> {
    let fit = Development {
        average,
        n_periods: Some(n),
        ..Development::default()
    }
    .fit(tri, column)
    .unwrap();
    fit.ldf
        .iter()
        .map(|f| (f * 1000.0).round() / 1000.0)
        .collect()
}

/// The cells of `column` as origin by age rows.
fn cells(tri: &Triangle, column: &str) -> Vec<Vec<Option<f64>>> {
    let c = tri.column_position(column).unwrap();
    let [_, _, no, nd] = tri.shape();
    (0..no)
        .map(|o| (0..nd).map(|d| tri.get(0, c, o, d)).collect())
        .collect()
}

// ---------------------------------------------------------------------------
// Chapter 11: frequency-severity
// ---------------------------------------------------------------------------

/// XYZ Insurer's ultimate counts of Exhibit II, Sheet 3: the closed and
/// reported count chain ladders on the printed factors to ultimate, their
/// mean selected.
fn xyz_counts() -> (UltimateCounts, UltimateCounts) {
    let closed = UltimateCounts::Development {
        column: "closed_counts".into(),
        method: printed_cdf(
            &[
                6.085, 2.281, 1.612, 1.318, 1.161, 1.085, 1.035, 1.021, 1.015, 1.003, 1.000,
            ],
            12,
        ),
    };
    let reported = UltimateCounts::Development {
        column: "reported_counts".into(),
        method: printed_cdf(
            &[
                1.131, 1.035, 1.013, 1.005, 1.002, 1.001, 1.000, 1.000, 1.000, 1.000, 1.000,
            ],
            12,
        ),
    };
    (closed, reported)
}

/// Chapter 11, Exhibit II: approach #1 on XYZ Insurer. The latest-2
/// volume-weighted averages of the counts and the ultimate counts, then the
/// reported severities developed with the printed factors to ultimate.
#[test]
fn friedland_11_xyz_frequency_severity_approach_1() {
    let tri = xyz();
    // Sheets 1 and 2, part 3: volume-weighted, latest 2.
    let closed = averages(&tri, "closed_counts", Average::Volume, 2);
    assert_eq!(
        closed,
        [
            2.668, 1.415, 1.223, 1.135, 1.070, 1.048, 1.014, 1.006, 1.012, 1.003
        ]
    );
    let reported = averages(&tri, "reported_counts", Average::Volume, 2);
    assert_eq!(
        reported,
        [
            1.092, 1.022, 1.008, 1.003, 1.001, 1.001, 1.000, 1.000, 1.000, 1.000
        ]
    );
    // Sheet 3.
    let (closed, reported) = xyz_counts();
    let selected = UltimateCounts::Mean(vec![closed.clone(), reported.clone()]);
    let closed = closed.resolve(&tri).unwrap();
    let reported = reported.resolve(&tri).unwrap();
    let selected_values = selected.resolve(&tri).unwrap();
    close_all(
        &closed,
        &[
            637., 1047., 1423., 1476., 1576., 1701., 2356., 2565., 1936., 1804., 1679.,
        ],
        0.5,
        "closed count ultimates",
    );
    close_all(
        &reported,
        &[
            637., 1047., 1408., 1455., 1554., 1631., 2263., 2402., 1679., 1308., 1172.,
        ],
        0.5,
        "reported count ultimates",
    );
    close(
        selected_values.iter().sum(),
        17378.0,
        1.0,
        "selected ultimate counts",
    );

    // Sheets 4 to 6: severities are reported claims ($000) over reported
    // counts, developed with the printed factors to ultimate.
    let sev = printed_cdf(
        &[
            2.310, 1.498, 1.212, 1.100, 1.066, 1.016, 1.003, 0.992, 0.990, 0.999, 1.000,
        ],
        12,
    );
    let fit = FrequencySeverity {
        amounts: "reported".into(),
        counts: "reported_counts".into(),
        ultimate_counts: CountProjection::Counts(selected),
        severity: SeverityProjection::Development(sev),
    }
    .fit(&tri)
    .unwrap();
    // Severities in dollars: the claims are in $000.
    let latest: Vec<f64> = fit.latest_severity.iter().map(|s| s * 1000.0).collect();
    close_all(
        &latest,
        &[
            24839., 23980., 26453., 26665., 30997., 27239., 31129., 29563., 29453., 25104., 17985.,
        ],
        1.5,
        "latest reported severities",
    );
    let ult: Vec<f64> = fit.ultimate_severity.iter().map(|s| s * 1000.0).collect();
    close_all(
        &ult,
        &[
            24839., 23956., 26189., 26452., 31090., 27675., 33183., 32519., 35697., 37606., 41544.,
        ],
        2.0,
        "ultimate severities",
    );
    // The data is printed rounded (counts and claims carry decimals in the
    // book's spreadsheet): severities agree within 2 dollars, ultimates
    // within 0.04%.
    let want = [
        15822., 25082., 37083., 38778., 48655., 46107., 76620., 80745., 64505., 58516., 59242.,
    ];
    for (k, (g, w)) in fit.ultimate.iter().zip(want).enumerate() {
        close(*g, w, 4e-4 * w, &format!("ultimate[{k}]"));
    }
    close(fit.total_ultimate(), 551155.0, 30.0, "total ultimate");
}

/// Chapter 11, Exhibit IV: approach #2 on XYZ Insurer, for 2007 and 2008:
/// a 2.36% frequency at the 2008 level on earned premium brought to 2008
/// rates (2007's on-level factor 0.800) with a -1.5% count trend, and a
/// $26,720 severity at the 2008 level with a 5% trend.
#[test]
fn friedland_11_xyz_frequency_severity_approach_2() {
    let tri = xyz();
    let base = Period::year(2008);
    let mut on_level = vec![1.0; 11];
    on_level[9] = 0.8;
    let fit = FrequencySeverity {
        amounts: "reported".into(),
        counts: "reported_counts".into(),
        ultimate_counts: CountProjection::Frequency {
            exposure: "premium".into(),
            exposure_factors: on_level,
            frequency: Trended::new(0.0236, base, -0.015),
        },
        // $000 per claim.
        severity: SeverityProjection::Trended(Trended::new(26.720, base, 0.05)),
    }
    .fit(&tri)
    .unwrap();
    // Sheet 1, line (11), and sheet 2, line (8).
    close(
        fit.ultimate_counts[9] / 62438.0,
        0.0192,
        5e-5,
        "2007 frequency",
    );
    close(
        fit.ultimate_severity[9] * 1000.0,
        25448.0,
        0.5,
        "2007 severity",
    );
    // Sheet 3, lines (3) and (5): the book multiplies the premium by the
    // frequency rounded to 1.92% (1,199 claims, 1,196.8 unrounded) and the
    // rounded counts by the rounded severities, 0.2% above the unrounded
    // product in 2007.
    close(
        fit.ultimate_counts[9],
        62438.0 * 0.0236 * 0.8 / 0.985,
        1e-9,
        "2007 counts",
    );
    close(fit.ultimate_counts[10], 1128.0, 0.5, "2008 counts");
    close(
        fit.ultimate[9],
        30512.152,
        0.0025 * 30512.152,
        "2007 ultimate",
    );
    close(fit.ultimate[10], 30140.260, 0.5, "2008 ultimate");
    // Line (8): IBNR.
    close(fit.reserves()[10], 11508.260, 0.5, "2008 IBNR");
}

/// The WC Self-Insurer of chapter 11, Exhibit III, sheet 10: a 0.36%
/// frequency at the 2008 level per $100 of payroll, which inflates 2.5% a
/// year, and a -1% count trend.
#[test]
fn friedland_11_wc_trended_frequency() {
    let t = Trended {
        selected: 0.0036,
        base: Period::year(2008),
        trend: -0.01,
        factors: vec![1.0 / 1.025, 1.0],
    };
    let f = t.values(&[Period::year(2007), Period::year(2008)]).unwrap();
    // 0.36% x 1.025 / 0.99 = 0.3727%, printed 0.37%.
    close(f[0], 0.0036 * 1.025 / 0.99, 1e-15, "2007 frequency");
    close(f[0] * 780_000.0, 2907.0, 0.5, "2007 counts");
    close(f[1] * 740_000.0, 2664.0, 0.5, "2008 counts");
}

/// Chapter 11, Exhibit VI: the disposal rate method on XYZ Insurer,
/// accident years 2001 to 2008 to 96 months. Disposal rates over the
/// reported count ultimates, selected by the simple average of the latest
/// 2; severities trended at 5% to 2008 with the tort reform factors
/// (0.670 to 2005, 0.750 in 2006) and selected the same way, the pooled
/// severity of 84 months and older beyond.
#[test]
fn friedland_11_xyz_disposal() {
    let tri = load("friedland_xyz", XYZ, 2001, 2008, 96);
    let reported = [1455., 1554., 1629., 2258., 2390., 1657., 1264., 1036.];
    let cdf = [1.000, 1.000, 1.001, 1.002, 1.005, 1.013, 1.035, 1.131];
    let ultimate: Vec<f64> = reported.iter().zip(cdf).map(|(r, c)| r * c).collect();
    let mut factors = vec![0.670; 5];
    factors.extend([0.750, 1.0, 1.0]);
    let method = Disposal {
        closed_counts: "closed_counts".into(),
        paid: "paid".into(),
        ultimate_counts: UltimateCounts::Given(ultimate),
        disposal_rates: Selection::latest(2),
        severities: Selection::latest(2),
        tail_severity_from: Some(84),
        trend: 0.05,
        base: Some(Period::year(2008)),
        factors,
    };
    let fit = method.fit(&tri).unwrap();
    // Sheet 1, part 3.
    let printed = [0.244, 0.572, 0.704, 0.814, 0.910, 0.952, 0.982, 0.994];
    let rates: Vec<f64> = fit
        .selected_rates
        .iter()
        .map(|r| (r * 1000.0).round() / 1000.0)
        .collect();
    assert_eq!(rates, printed);
    // The book projects with the rates rounded to three decimals.
    let fit = Disposal {
        disposal_rates: Selection::Given(
            printed
                .iter()
                .enumerate()
                .map(|(k, &r)| (12 + 12 * k as Lag, r))
                .collect(),
        ),
        ..method
    }
    .fit(&tri)
    .unwrap();
    // Sheet 2: projected incremental closed counts, step to ultimate last.
    let counts_2008 = [276., 389., 156., 130., 114., 50., 36., 14., 7.];
    let got: Vec<f64> = fit.incremental_counts[7]
        .iter()
        .map(|c| c.unwrap())
        .collect();
    close_all(&got, &counts_2008, 0.5, "2008 incremental closed counts");
    // Sheets 4 and 5, in dollars. The claims are rounded to the $000, which
    // moves the selections by a few dollars (the pooled tail severity by 8).
    let sel: Vec<f64> = fit.selected_severities.iter().map(|s| s * 1000.0).collect();
    close_all(
        &sel,
        &[
            11807., 15165., 26043., 35183., 41908., 62206., 70432., 70432., 70432.,
        ],
        10.0,
        "selected severities at the 2008 level",
    );
    // Sheet 6: 2005 at 60 months is 41,908 / 1.05^3 / 0.670.
    close(
        fit.projected_severities[4][4].unwrap() * 1000.0,
        54032.0,
        5.0,
        "2005 severity at 60 months",
    );
    // Sheet 8: ultimates and unpaid. The printed incremental cells are
    // rounded, so the totals agree to about 6 thousand dollars in 386 million.
    close_all(
        &fit.ultimate,
        &[
            39192., 46869., 44479., 71906., 71684., 49913., 31805., 29828.,
        ],
        2.0,
        "ultimates",
    );
    close(fit.total_ultimate(), 385676.0, 6.0, "total ultimate");
    close(
        fit.total_reserve(),
        385676.0 - 253208.0,
        6.0,
        "total unpaid",
    );
}

// ---------------------------------------------------------------------------
// Chapter 12: case outstanding development
// ---------------------------------------------------------------------------

/// The printed selections at 24 months and older as a given selection.
fn given(values: &[f64]) -> Selection {
    Selection::Given(
        values
            .iter()
            .enumerate()
            .map(|(k, &v)| (24 + 12 * k as Lag, v))
            .collect(),
    )
}

/// Chapter 12, Exhibit I: U.S. Industry Auto, ratios selected by the simple
/// average of the latest 3, 1.10 paid to ultimate.
#[test]
fn friedland_12_us_auto_case_outstanding() {
    let tri = load("friedland_us_auto", &["reported", "paid"], 1998, 2007, 120);
    let method = CaseOutstanding {
        paid: "paid".into(),
        incurred: "reported".into(),
        paid_on_case: Selection::latest(3),
        case_on_case: Selection::latest(3),
        paid_to_ultimate: 1.10,
    };
    let fit = method.fit(&tri).unwrap();
    // Sheets 2 and 3: the selected ratios at 24 to 120 months.
    let round = |v: &[f64]| -> Vec<f64> {
        v[1..]
            .iter()
            .map(|r| (r * 1000.0).round() / 1000.0)
            .collect()
    };
    let paid_on_case = [
        0.833, 0.701, 0.714, 0.714, 0.653, 0.631, 0.553, 0.437, 0.524,
    ];
    let case_on_case = [
        0.526, 0.566, 0.528, 0.486, 0.511, 0.555, 0.632, 0.674, 0.580,
    ];
    assert_eq!(round(&fit.selected_paid_on_case), paid_on_case);
    assert_eq!(round(&fit.selected_case_on_case), case_on_case);
    // Sheets 4 and 5: the book projects with the ratios rounded to three
    // decimals; with them the ultimates agree to the unit ($000).
    let fit = CaseOutstanding {
        paid_on_case: given(&paid_on_case),
        case_on_case: given(&case_on_case),
        ..method
    }
    .fit(&tri)
    .unwrap();
    let want = [
        47752116., 51215775., 54905021., 56436079., 58880132., 58139086., 58163235., 59474287.,
        60321169., 62470839.,
    ];
    close_all(&fit.ultimate, &want, 1.0, "ultimates");
    close(fit.total_ultimate(), 567757738.0, 2.0, "total ultimate");
    close(fit.total_reserve(), 69707370.0, 2.0, "total unpaid");
    close(fit.ibnr().iter().sum(), 24276151.0, 2.0, "total IBNR");
    // The 2007 case outstanding at 24 months: 0.526 x 21,623,594.
    close(
        fit.case_outstanding[9][1].unwrap(),
        11374010.0,
        1.0,
        "2007 case at 24",
    );
}

/// Chapter 12, Exhibit II: XYZ Insurer, ratios selected by the simple
/// average of the latest 2, 1.10 paid to ultimate.
///
/// The book's Sheet 4 leaves out the payment of 2000 at 120 months (0.501
/// times its case outstanding of 465 at 108, 233): it prints 0 there and
/// projects the payment at 132 on the case outstanding at 120 as every
/// other row does. Its 2000 ultimate is 233 short of its own method.
#[test]
fn friedland_12_xyz_case_outstanding() {
    let paid_on_case = [
        0.437, 0.369, 0.415, 0.494, 0.596, 0.730, 0.852, 0.618, 0.501, 0.816,
    ];
    let case_on_case = [
        1.367, 0.986, 0.783, 0.584, 0.583, 0.353, 0.310, 0.514, 0.344, 0.000,
    ];
    let fit = CaseOutstanding {
        paid: "paid".into(),
        incurred: "reported".into(),
        paid_on_case: Selection::latest(2),
        case_on_case: Selection::latest(2),
        paid_to_ultimate: 1.10,
    }
    .fit(&xyz())
    .unwrap();
    // The data is rounded to the $000, so the 132-month ratio is 0.817 on it
    // (58 / 71) and 0.816 in the book.
    close_all(
        &fit.selected_paid_on_case[1..],
        &paid_on_case,
        1.5e-3,
        "paid on case",
    );
    close_all(
        &fit.selected_case_on_case[1..],
        &case_on_case,
        1.5e-3,
        "case on case",
    );
    // Unlike Exhibit I, the book projects XYZ with its unrounded ratios.
    let mut want = [
        15822., 25054., 36912., 38803., 48796., 45093., 74873., 77723., 58665., 46198., 46005.,
    ];
    let missing = 0.501 * 465.0;
    want[2] += missing;
    // The printed data is rounded to the $000: within 2 per origin.
    close_all(&fit.ultimate, &want, 2.0, "ultimates");
    close(
        fit.total_ultimate(),
        513944.0 + missing,
        6.0,
        "total ultimate",
    );
}

/// Chapter 12, Exhibit III: the case development factor from reported and
/// paid factors to ultimate.
#[test]
fn friedland_12_case_development_factor() {
    let reported = [1.015, 1.020, 1.030, 1.051, 1.077, 1.131];
    let paid = [1.046, 1.067, 1.109, 1.187, 1.306, 1.489];
    let printed = [1.506, 1.454, 1.421, 1.445, 1.439, 1.545];
    let case = [500_000., 650_000., 800_000., 850_000., 975_000., 1_000_000.];
    let mut total = 0.0;
    for k in 0..6 {
        let f = case_development_factor(reported[k], paid[k]).unwrap();
        close(f, printed[k], 5e-4, "case factor");
        total += case[k] * (f * 1000.0).round() / 1000.0;
    }
    close(total, 7011175.0, 1e-6, "unpaid estimate");
}

// ---------------------------------------------------------------------------
// Chapter 13: Berquist-Sherman
// ---------------------------------------------------------------------------

fn berqsherm(name: &str) -> Triangle {
    load(
        name,
        &["incurred", "paid", "reported", "closed"],
        1969,
        1976,
        96,
    )
}

fn reported_less_closed() -> OpenCounts {
    OpenCounts::ReportedLessClosed {
        reported: "reported".into(),
        closed: "closed".into(),
    }
}

/// Chapter 13, Exhibit I: Berquist and Sherman's medical malpractice
/// insurer, the case reserve adequacy adjustment at a 15% trend.
#[test]
fn friedland_13_medmal_case_adequacy() {
    let tri = berqsherm("berqsherm_medmal");
    let case = CaseAdequacy {
        incurred: "incurred".into(),
        paid: "paid".into(),
        open_counts: reported_less_closed(),
        trend: CaseTrend::Given(0.15),
        max_age: None,
    };
    // Sheet 4: the trend of the average case outstanding by age.
    let avg = case.average_case(&tri).unwrap();
    let trends = trend_by_age(&avg, "average_case").unwrap();
    let rate: Vec<f64> = trends[..7]
        .iter()
        .map(|(_, t)| (t.unwrap().rate * 1000.0).round() / 10.0)
        .collect();
    assert_eq!(rate, [15.6, 29.5, 31.1, 34.2, 33.0, 32.2, 27.6]);
    let r2: Vec<f64> = trends[..7]
        .iter()
        .map(|(_, t)| (t.unwrap().r_squared * 1000.0).round() / 10.0)
        .collect();
    assert_eq!(r2, [80.0, 89.5, 85.8, 94.1, 98.9, 98.3, 100.0]);
    // Sheet 6: the adjusted reported claims, to the dollar.
    let adjusted = case.adjust(&tri).unwrap();
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
    let got = cells(&adjusted, "incurred");
    for (o, row) in want.iter().enumerate() {
        for (d, w) in row.iter().enumerate() {
            close(
                got[o][d].unwrap(),
                *w,
                0.5,
                &format!("adjusted incurred {o} {d}"),
            );
        }
    }
    // Sheet 7: the simple average factors of all years.
    let fit = Development {
        average: Average::Simple,
        ..Development::default()
    }
    .fit(&adjusted, "incurred")
    .unwrap();
    let ldf: Vec<f64> = fit
        .ldf
        .iter()
        .map(|f| (f * 1000.0).round() / 1000.0)
        .collect();
    assert_eq!(ldf, [3.906, 1.534, 1.340, 0.925, 1.065, 0.964, 0.979]);
    // Sheet 8: the printed factors to ultimate give 521,181,558.
    let cl = printed_cdf(
        &[7.465, 1.911, 1.246, 0.930, 1.005, 0.944, 0.979, 1.000],
        12,
    )
    .fit(&adjusted, "incurred")
    .unwrap();
    close(
        cl.total_ultimate(),
        521181558.0,
        1.0,
        "adjusted reported ultimate",
    );
}

/// Berquist and Sherman's ultimate counts in Friedland, chapter 13, Exhibit
/// II, sheet 4: the latest reported counts times the printed factors to
/// ultimate.
fn auto_bi_ultimate_counts(tri: &Triangle) -> UltimateCounts {
    let cdf = [1.000, 1.000, 1.000, 1.001, 1.003, 1.008, 1.021, 1.221];
    let latest = tri.latest_diagonal();
    let reported = latest.values(0, tri.column_position("reported").unwrap());
    UltimateCounts::Given(reported.iter().zip(cdf).map(|(r, c)| r * c).collect())
}

/// Chapter 13, Exhibit II: Berquist and Sherman's automobile bodily injury
/// insurer, the settlement rate adjustment.
#[test]
fn friedland_13_auto_bi_settlement_rate() {
    let tri = berqsherm("berqsherm_auto");
    let adjusted = SettlementRate {
        paid: "paid".into(),
        closed_counts: "closed".into(),
        ultimate_counts: auto_bi_ultimate_counts(&tri),
        disposal_rates: Selection::LatestDiagonal,
        interpolation: Interpolation::Exponential,
    }
    .adjust(&tri)
    .unwrap();
    // Sheet 7: adjusted closed counts and paid claims ($000), printed to
    // the unit.
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
    let got_closed = cells(&adjusted, "closed");
    let got_paid = cells(&adjusted, "paid");
    for o in 0..8 {
        for d in 0..closed[o].len() {
            close(
                got_closed[o][d].unwrap(),
                closed[o][d],
                0.5,
                &format!("closed {o} {d}"),
            );
            close(
                got_paid[o][d].unwrap(),
                paid[o][d],
                0.5,
                &format!("paid {o} {d}"),
            );
        }
    }
    // Sheet 8: volume-weighted factors of all years.
    let fit = ChainLadder::default().fit(&adjusted, "paid").unwrap();
    let ldf: Vec<f64> = fit
        .development
        .ldf
        .iter()
        .map(|f| (f * 1000.0).round() / 1000.0)
        .collect();
    assert_eq!(ldf, [3.158, 1.538, 1.277, 1.114, 1.047, 1.018, 1.006]);
    // Sheet 11, column (9): the volume-weighted projection of the adjusted
    // paid claims, in thousands, on the printed cells.
    close(
        fit.total_ultimate(),
        132019.0,
        15.0,
        "adjusted paid ultimate",
    );
}

/// Chapter 13, Exhibit III: XYZ Insurer. The average paid claim trends
/// (sheet 1), the case adjustment at 5% for ages to 84 months (sheets 2 and
/// 3), the settlement adjustment on the reported count ultimates of chapter
/// 11 (sheets 4 to 7) and both (sheets 8 to 10).
#[test]
fn friedland_13_xyz() {
    let tri = xyz();
    // Sheet 1: average paid on closed counts, the trend by age.
    let avg = ratio_column(&tri, "paid", "closed_counts", "average_paid").unwrap();
    let trends = trend_by_age(&avg, "average_paid").unwrap();
    let rate: Vec<f64> = trends[..10]
        .iter()
        .map(|(_, t)| (t.unwrap().rate * 1000.0).round() / 10.0)
        .collect();
    assert_eq!(rate, [8.1, 5.4, 4.6, 4.3, 5.5, 5.1, 6.8, 4.8, 3.1, -4.2]);

    let case = CaseAdequacy {
        incurred: "reported".into(),
        paid: "paid".into(),
        open_counts: OpenCounts::ReportedLessClosed {
            reported: "reported_counts".into(),
            closed: "closed_counts".into(),
        },
        trend: CaseTrend::Given(0.05),
        max_age: Some(84),
    };
    // Sheet 2: case adjusted reported claims ($000). The book's counts carry
    // decimals, so cells agree within 3.
    let case_adjusted = case.adjust(&tri).unwrap();
    let got = cells(&case_adjusted, "reported");
    let rows_2001 = [
        15789., 29146., 35224., 39380., 39748., 38452., 39706., 38798.,
    ];
    for (d, w) in rows_2001.iter().enumerate() {
        close(
            got[3][d].unwrap(),
            *w,
            3.0,
            &format!("case adjusted 2001 at {}", 12 * (d + 1)),
        );
    }
    let row_1998 = [
        14600., 15094., 15513., 17104., 16366., 16163., 15835., 15822.,
    ];
    for (d, w) in row_1998.iter().enumerate() {
        close(
            got[0][d + 3].unwrap(),
            *w,
            3.0,
            &format!("case adjusted 1998 at {}", 48 + 12 * d),
        );
    }
    assert!(got[0][2].is_none(), "1998 at 36 months has no counts");
    // Sheet 3: latest-3 volume-weighted averages, 1.658, 1.159, 1.040.
    let avg3 = averages(&case_adjusted, "reported", Average::Volume, 3);
    assert_eq!(&avg3[..3], [1.658, 1.159, 1.040]);
    // Sheet 10, column (8): 476,678.
    let cl = printed_cdf(
        &[1.998, 1.205, 1.040, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        12,
    )
    .fit(&case_adjusted, "reported")
    .unwrap();
    close(cl.total_ultimate(), 476678.0, 3.0, "case adjusted ultimate");

    // Sheets 4 to 7: the settlement adjustment on chapter 11's reported
    // count ultimates.
    let (_, reported) = xyz_counts();
    let settlement = SettlementRate {
        paid: "paid".into(),
        closed_counts: "closed_counts".into(),
        ultimate_counts: reported,
        disposal_rates: Selection::LatestDiagonal,
        interpolation: Interpolation::Exponential,
    };
    let paid_adjusted = settlement.adjust(&tri).unwrap();
    let closed = cells(&paid_adjusted, "closed_counts");
    let paid = cells(&paid_adjusted, "paid");
    // 1998 is filled from 12 months, before its first count.
    let closed_1998 = [
        150., 385., 456., 516., 571., 613., 624., 633., 634., 635., 637.,
    ];
    let paid_1998 = [
        1658., 4830., 6659., 8760., 11401., 14476., 15439., 15705., 15742., 15769., 15822.,
    ];
    let paid_2005 = [4805., 19876., 29202., 40026.];
    for d in 0..11 {
        close(
            closed[0][d].unwrap(),
            closed_1998[d],
            0.5,
            &format!("closed 1998 {d}"),
        );
        close(
            paid[0][d].unwrap(),
            paid_1998[d],
            2.0,
            &format!("paid 1998 {d}"),
        );
    }
    for d in 0..4 {
        close(
            paid[7][d].unwrap(),
            paid_2005[d],
            2.0,
            &format!("paid 2005 {d}"),
        );
    }
    // Sheet 7: volume-weighted latest 5.
    let avg5 = averages(&paid_adjusted, "paid", Average::Volume, 5);
    assert_eq!(
        avg5,
        [
            4.152, 1.432, 1.322, 1.269, 1.206, 1.053, 1.044, 1.005, 1.002, 1.003
        ]
    );
    // Sheet 10, column (10): 485,546 with the printed factors and a 1.010
    // tail.
    let cl = printed_cdf(
        &[
            13.490, 3.249, 2.269, 1.716, 1.352, 1.121, 1.065, 1.020, 1.015, 1.013, 1.010,
        ],
        12,
    )
    .fit(&paid_adjusted, "paid")
    .unwrap();
    close(
        cl.total_ultimate(),
        485546.0,
        3.0,
        "settlement adjusted ultimate",
    );

    // Sheets 8 and 9: both adjustments, the average case to 84 months as
    // above and the settlement adjustment at every age.
    let both = BerquistSherman {
        case_adequacy: CaseAdequacy {
            max_age: Some(84),
            ..case
        },
        settlement_rate: settlement,
    }
    .adjust(&tri)
    .unwrap();
    let got = cells(&both, "reported");
    let row_2001 = [
        15467., 27457., 34478., 38046., 38343., 38516., 39877., 38798.,
    ];
    for (d, w) in row_2001.iter().enumerate() {
        close(
            got[3][d].unwrap(),
            *w,
            3.0,
            &format!("both adjusted 2001 at {}", 12 * (d + 1)),
        );
    }
    let row_1998 = [
        14541., 15031., 15934., 16697., 16012., 15878., 15834., 15822.,
    ];
    for (d, w) in row_1998.iter().enumerate() {
        close(
            got[0][d + 3].unwrap(),
            *w,
            3.0,
            &format!("both adjusted 1998 at {}", 48 + 12 * d),
        );
    }
    // Sheet 10, column (9): 489,258.
    let cl = printed_cdf(
        &[
            2.229, 1.345, 1.094, 1.010, 1.007, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        ],
        12,
    )
    .fit(&both, "reported")
    .unwrap();
    close(cl.total_ultimate(), 489258.0, 3.0, "both adjusted ultimate");
}

// ---------------------------------------------------------------------------
// chainladder-python
// ---------------------------------------------------------------------------

/// chainladder-python's `BerquistSherman` on its `berqsherm` sample:
/// incurred from the case adjustment alone (its open counts are reported
/// less the unadjusted closed counts, and it adds the unadjusted paid),
/// paid and closed counts from the settlement adjustment on the chain
/// ladder's ultimate reported counts.
#[test]
fn chainladder_python_berquist_sherman() {
    let cases = reference("reserving_berquist_sherman_python.csv");
    let mut cache: BTreeMap<(String, String), Triangle> = BTreeMap::new();
    check(&cases, |case| {
        let dataset = case.get("dataset").to_string();
        let method = case.get("method").to_string();
        let tri = cache
            .entry((dataset.clone(), method.clone()))
            .or_insert_with(|| {
                let tri = berqsherm(&dataset);
                let trend = case.param("method", "trend");
                let incurred = CaseAdequacy {
                    incurred: "incurred".into(),
                    paid: "paid".into(),
                    open_counts: reported_less_closed(),
                    trend: CaseTrend::Given(trend),
                    max_age: None,
                }
                .adjust(&tri)
                .unwrap();
                let settled = SettlementRate {
                    paid: "paid".into(),
                    closed_counts: "closed".into(),
                    ultimate_counts: UltimateCounts::development("reported"),
                    disposal_rates: Selection::LatestDiagonal,
                    interpolation: Interpolation::Exponential,
                }
                .adjust(&tri)
                .unwrap();
                let rows = cells(&incurred, "incurred");
                let paid = cells(&settled, "paid");
                let closed = cells(&settled, "closed");
                // Assemble the three adjusted columns on one triangle.
                rebuild(
                    &tri,
                    &[("incurred", rows), ("paid", paid), ("closed", closed)],
                )
            });
        let quantity = case.get("quantity");
        let arg = case.get("arg");
        let origin_index = |year: i32| (year - 1969) as usize;
        if let Some(column) = quantity.strip_suffix("_ultimate") {
            let fit = ChainLadder::default().fit(tri, column).ok()?;
            Some(fit.ultimate[origin_index(arg.parse().ok()?)])
        } else {
            let (o, a) = arg.split_once(':')?;
            let (o, a): (i32, Lag) = (o.parse().ok()?, a.parse().ok()?);
            let c = tri.column_position(quantity).ok()?;
            tri.get(0, c, origin_index(o), (a / 12 - 1) as usize)
        }
    });
}

/// A triangle on `like`'s grid with the given columns of rows.
fn rebuild(like: &Triangle, columns: &[(&str, Vec<Vec<Option<f64>>>)]) -> Triangle {
    let origins = like.origins();
    let ages = like.development();
    let mut origin = Vec::new();
    let mut age = Vec::new();
    let mut values: Vec<Vec<f64>> = vec![Vec::new(); columns.len()];
    for (o, p) in origins.iter().enumerate() {
        for (d, &a) in ages.iter().enumerate() {
            if columns.iter().all(|(_, r)| r[o][d].is_none()) {
                continue;
            }
            origin.push(p.start());
            age.push(a);
            for (k, (_, r)) in columns.iter().enumerate() {
                values[k].push(r[o][d].unwrap_or(f64::NAN));
            }
        }
    }
    let values: Vec<(&str, &[f64])> = columns
        .iter()
        .zip(&values)
        .map(|((n, _), v)| (*n, v.as_slice()))
        .collect();
    Triangle::from_long(&Long {
        keys: &[],
        origin: &origin,
        development: DevelopmentColumn::Age(&age),
        values: &values,
        origin_grain: Grain::Year,
        development_grain: Grain::Year,
        cumulative: true,
    })
    .unwrap()
}

// ---------------------------------------------------------------------------
// Chapter 11, GL Insurer
// ---------------------------------------------------------------------------

/// Chapter 11, Exhibit V: the disposal rate method on GL Insurer, the
/// chapter's own worked example and the only one that selects its rates by
/// the medial average of the latest 5, and pools the tail from an age before
/// the oldest two. Sheet 3's selected counts are the mean of the closed and
/// reported projections on the printed factors; the printed counts behind
/// them are the only input besides the two triangles.
#[test]
fn friedland_11_gl_disposal() {
    let tri = load("friedland_gl", &["closed_counts", "paid"], 2001, 2008, 96);
    let closed = [796., 626., 542., 494., 403., 300., 188., 127.];
    let closed_cdf = [1.100, 1.123, 1.174, 1.256, 1.390, 1.682, 2.199, 4.769];
    let reported = [870., 731., 636., 684., 687., 734., 620., 812.];
    let reported_cdf = [1.000, 1.007, 0.968, 0.932, 0.897, 0.818, 0.746, 0.753];
    let ultimate: Vec<f64> = (0..8)
        .map(|o| (closed[o] * closed_cdf[o] + reported[o] * reported_cdf[o]) / 2.0)
        .collect();
    close_all(
        &ultimate,
        &[873., 720., 626., 629., 588., 553., 438., 609.],
        0.5,
        "selected ultimate counts",
    );
    let method = Disposal {
        closed_counts: "closed_counts".into(),
        paid: "paid".into(),
        ultimate_counts: UltimateCounts::Given(ultimate),
        disposal_rates: Selection::Average {
            n_periods: Some(5),
            drop_high: 1,
            drop_low: 1,
        },
        severities: Selection::latest(3),
        tail_severity_from: Some(72),
        trend: 0.05,
        base: Some(Period::year(2008)),
        factors: Vec::new(),
    };
    let fit = method.fit(&tri).unwrap();
    // Sheet 4, part 3: the medial averages, the 72-month one of three values
    // and the 84-month one of two.
    let rates: Vec<f64> = fit
        .selected_rates
        .iter()
        .map(|r| (r * 1000.0).round() / 1000.0)
        .collect();
    assert_eq!(
        rates,
        [0.200, 0.433, 0.585, 0.710, 0.791, 0.862, 0.882, 0.912]
    );
    // The book projects with the rates rounded to three decimals.
    let fit = Disposal {
        disposal_rates: Selection::Given(
            rates
                .iter()
                .enumerate()
                .map(|(k, &r)| (12 + 12 * k as Lag, r))
                .collect(),
        ),
        ..method
    }
    .fit(&tri)
    .unwrap();
    // Sheet 6: the incremental paid severities.
    let row_2002 = [7095., 32500., 53717., 64710., 105000., 229511., 126828.];
    for (d, w) in row_2002.iter().enumerate() {
        close(
            fit.severities[1][d].unwrap(),
            *w,
            0.5,
            &format!("2002 severity at {}", 12 * (d + 1)),
        );
    }
    // Sheets 8 and 9: the selected severities at the 2008 level, the pooled
    // tail from 72 months on carried to the step to ultimate.
    close_all(
        &fit.selected_severities,
        &[
            11259., 32980., 65523., 80544., 140802., 175816., 175816., 175816., 175816.,
        ],
        0.5,
        "selected severities",
    );
    // Sheet 10: 2002 at 96 months is 175,816 / 1.05^6.
    close(
        fit.projected_severities[1][7].unwrap(),
        131196.0,
        0.5,
        "2002 severity at 96",
    );
    // Sheet 5: the 2008 row and the step to ultimate of each origin.
    let got: Vec<f64> = fit.incremental_counts[7]
        .iter()
        .map(|c| c.unwrap())
        .collect();
    close_all(
        &got,
        &[127., 140., 91., 75., 49., 43., 12., 18., 53.],
        0.5,
        "2008 incremental closed counts",
    );
    // Sheets 11 and 12, in dollars: every origin's ultimate. The book's counts
    // and selections carry hidden decimals, worth up to 30 dollars in 40 million.
    close_all(
        &fit.ultimate,
        &[
            39497433., 44743308., 35510981., 37766027., 42007442., 41113459., 32859080., 47109641.,
        ],
        100.0,
        "ultimates",
    );
    close(
        fit.total_ultimate() / 1000.0,
        320607.0,
        0.5,
        "total ultimate",
    );
    close(fit.total_reserve() / 1000.0, 185945.0, 0.5, "total unpaid");
}

// ---------------------------------------------------------------------------
// The choices the exhibits do not reach
// ---------------------------------------------------------------------------

/// A cumulative annual triangle from rows of columns, each row from 12 months.
fn small(columns: &[(&str, &[&[f64]])]) -> Triangle {
    let mut origin = Vec::new();
    let mut age: Vec<Lag> = Vec::new();
    for (o, r) in columns[0].1.iter().enumerate() {
        for d in 0..r.len() {
            origin.push(Month::january(2020 + o as i32));
            age.push(12 * (d as Lag + 1));
        }
    }
    let values: Vec<(&str, Vec<f64>)> = columns
        .iter()
        .map(|(n, rows)| (*n, rows.iter().flat_map(|r| r.iter().copied()).collect()))
        .collect();
    let refs: Vec<(&str, &[f64])> = values.iter().map(|(n, v)| (*n, v.as_slice())).collect();
    Triangle::from_long(&Long {
        keys: &[],
        origin: &origin,
        development: DevelopmentColumn::Age(&age),
        values: &refs,
        origin_grain: Grain::Year,
        development_grain: Grain::Year,
        cumulative: true,
    })
    .unwrap()
}

fn medmal_case(trend: CaseTrend) -> CaseAdequacy {
    CaseAdequacy {
        incurred: "incurred".into(),
        paid: "paid".into(),
        open_counts: reported_less_closed(),
        trend,
        max_age: None,
    }
}

/// #554 asks for a trend fitted on the average case. On a triangle whose
/// averages grow exactly, each fit reads its own rate, and the average case's
/// rate leaves incurred as it is, since the latest diagonal detrended at the
/// true rate is the data. On the medical malpractice data the pooled trend
/// of the average case lies among the printed trends by age, and a fitted
/// trend restates exactly as the same rate given by hand.
#[test]
fn fitted_trends_read_their_rate_and_restate_as_given_rates() {
    let closed: [&[f64]; 3] = [&[10., 20., 30.], &[10., 20.], &[10.]];
    let open: [&[f64]; 3] = [&[4., 3., 2.], &[4., 3.], &[4.]];
    let grow = |counts: &[&[f64]; 3], level: &[f64; 3], rate: f64| -> Vec<Vec<f64>> {
        counts
            .iter()
            .enumerate()
            .map(|(o, r)| {
                r.iter()
                    .zip(level)
                    .map(|(n, a)| n * a * (1.0 + rate).powi(o as i32))
                    .collect()
            })
            .collect()
    };
    let paid = grow(&closed, &[5., 6., 7.], 0.1);
    let case = grow(&open, &[50., 60., 70.], 0.2);
    let incurred: Vec<Vec<f64>> = paid
        .iter()
        .zip(&case)
        .map(|(p, c)| p.iter().zip(c).map(|(p, c)| p + c).collect())
        .collect();
    let paid_refs: Vec<&[f64]> = paid.iter().map(Vec::as_slice).collect();
    let incurred_refs: Vec<&[f64]> = incurred.iter().map(Vec::as_slice).collect();
    let tri = small(&[
        ("incurred", &incurred_refs),
        ("paid", &paid_refs),
        ("closed", &closed),
        ("open", &open),
    ]);
    let exact = |trend: CaseTrend| {
        let case = CaseAdequacy {
            open_counts: OpenCounts::Column("open".into()),
            ..medmal_case(trend)
        };
        cells(&case.adjust(&tri).unwrap(), "incurred")
    };
    let observed = cells(&tri, "incurred");
    let restated = exact(CaseTrend::AverageCase);
    for (a, b) in observed.iter().flatten().zip(restated.iter().flatten()) {
        match (a, b) {
            (Some(a), Some(b)) => close(*a, *b, 1e-9, "the true rate changes nothing"),
            (a, b) => assert_eq!(a, b),
        }
    }
    let fitted = exact(CaseTrend::Fitted {
        amount: "paid".into(),
        counts: "closed".into(),
    });
    let given = exact(CaseTrend::Given(0.1));
    for (a, b) in fitted.iter().flatten().zip(given.iter().flatten()) {
        match (a, b) {
            (Some(a), Some(b)) => close(*a, *b, 1e-9, "paid over closed reads 10%"),
            (a, b) => assert_eq!(a, b),
        }
    }

    let tri = berqsherm("berqsherm_medmal");
    let on = |trend: CaseTrend| medmal_case(trend).adjust(&tri).unwrap();
    let average = medmal_case(CaseTrend::Given(0.0))
        .average_case(&tri)
        .unwrap();
    let case_rate = pooled_trend(&average, "average_case")
        .unwrap()
        .unwrap()
        .rate;
    assert!((0.156..=0.342).contains(&case_rate), "{case_rate}");
    assert_eq!(
        cells(&on(CaseTrend::AverageCase), "incurred"),
        cells(&on(CaseTrend::Given(case_rate)), "incurred")
    );
}

/// Open counts as a column are the same as reported less closed when the
/// column is their difference.
#[test]
fn an_open_count_column_is_reported_less_closed() {
    let incurred: [&[f64]; 3] = [&[100., 150., 160.], &[120., 170.], &[130.]];
    let paid: [&[f64]; 3] = [&[50., 100., 140.], &[60., 110.], &[70.]];
    let reported: [&[f64]; 3] = [&[10., 12., 12.], &[11., 13.], &[12.]];
    let closed: [&[f64]; 3] = [&[5., 9., 11.], &[5., 10.], &[6.]];
    let open: [&[f64]; 3] = [&[5., 3., 1.], &[6., 3.], &[6.]];
    let tri = small(&[
        ("incurred", &incurred),
        ("paid", &paid),
        ("reported", &reported),
        ("closed", &closed),
        ("open", &open),
    ]);
    let on = |open_counts| {
        let case = CaseAdequacy {
            open_counts,
            ..medmal_case(CaseTrend::Given(0.1))
        };
        cells(&case.adjust(&tri).unwrap(), "incurred")
    };
    let got = on(OpenCounts::Column("open".into()));
    assert_eq!(got, on(reported_less_closed()));
    // 2021 at 12 months: its 6 open at 2022's average, (130 - 70) / 6, a
    // year back, plus paid 60.
    close(
        got[1][0].unwrap(),
        60.0 + 6.0 * 10.0 / 1.1,
        1e-9,
        "2021 at 12",
    );
}

/// Both curves leave the latest diagonal alone and differ off it, on
/// Berquist and Sherman's data.
#[test]
fn linear_and_exponential_settlement_agree_on_the_latest_diagonal_only() {
    let tri = berqsherm("berqsherm_auto");
    let on = |interpolation| {
        SettlementRate {
            paid: "paid".into(),
            closed_counts: "closed".into(),
            ultimate_counts: auto_bi_ultimate_counts(&tri),
            disposal_rates: Selection::LatestDiagonal,
            interpolation,
        }
        .adjust(&tri)
        .unwrap()
    };
    let observed = cells(&tri, "paid");
    let exponential = on(Interpolation::Exponential);
    let linear = on(Interpolation::Linear);
    let (e, l) = (cells(&exponential, "paid"), cells(&linear, "paid"));
    let mut differ = 0;
    for o in 0..8 {
        let last = 7 - o;
        assert_eq!(e[o][last], observed[o][last], "exponential latest {o}");
        assert_eq!(l[o][last], observed[o][last], "linear latest {o}");
        for d in 0..last {
            if (e[o][d].unwrap() - l[o][d].unwrap()).abs() > 1e-6 {
                differ += 1;
            }
        }
    }
    assert!(differ > 15, "{differ} cells differ");
}

/// Cutting the oldest ages off a triangle leaves origins that stop at the
/// oldest age without being on the latest diagonal. The latest diagonal at
/// that age is still the most recent origin there, so the cells kept are
/// the same as on the whole triangle.
#[test]
fn the_latest_diagonal_of_a_triangle_cut_at_an_age_is_its_most_recent_origin() {
    let whole = xyz();
    let cut = load("friedland_xyz", XYZ, 1998, 2008, 96);
    let case = CaseAdequacy {
        incurred: "reported".into(),
        paid: "paid".into(),
        open_counts: OpenCounts::ReportedLessClosed {
            reported: "reported_counts".into(),
            closed: "closed_counts".into(),
        },
        trend: CaseTrend::Given(0.05),
        max_age: None,
    };
    let a = cells(&case.adjust(&whole).unwrap(), "reported");
    let b = cells(&case.adjust(&cut).unwrap(), "reported");
    for o in 0..11 {
        for d in 0..8 {
            match (a[o][d], b[o][d]) {
                (Some(x), Some(y)) => close(x, y, 1e-6, &format!("case {o} {d}")),
                (x, y) => assert_eq!(x, y, "case {o} {d}"),
            }
        }
    }
    let ultimate = UltimateCounts::Given(xyz_counts().1.resolve(&whole).unwrap());
    let settlement = SettlementRate {
        paid: "paid".into(),
        closed_counts: "closed_counts".into(),
        ultimate_counts: ultimate,
        disposal_rates: Selection::LatestDiagonal,
        interpolation: Interpolation::Exponential,
    };
    let a = cells(&settlement.adjust(&whole).unwrap(), "closed_counts");
    let b = cells(&settlement.adjust(&cut).unwrap(), "closed_counts");
    for o in 0..11 {
        for d in 0..8 {
            match (a[o][d], b[o][d]) {
                (Some(x), Some(y)) => close(x, y, 1e-6, &format!("closed {o} {d}")),
                (x, y) => assert_eq!(x, y, "closed {o} {d}"),
            }
        }
    }
}

/// Annual origins on quarterly ages: the latest diagonal has cells at 12, 24
/// and so on only, so the ages between have no adequacy to restate to.
#[test]
fn an_age_off_the_latest_diagonal_is_an_error_and_not_a_silent_zero() {
    let ages: Vec<Lag> = (1..=8)
        .map(|k| 3 * k)
        .chain((1..=4).map(|k| 3 * k))
        .collect();
    let origin: Vec<Month> = std::iter::repeat_n(Month::january(2020), 8)
        .chain(std::iter::repeat_n(Month::january(2021), 4))
        .collect();
    let v = |base: f64| -> Vec<f64> { ages.iter().map(|&a| base + f64::from(a)).collect() };
    let (incurred, paid, reported, closed) = (v(200.), v(100.), v(15.), v(5.));
    let tri = Triangle::from_long(&Long {
        keys: &[],
        origin: &origin,
        development: DevelopmentColumn::Age(&ages),
        values: &[
            ("incurred", &incurred),
            ("paid", &paid),
            ("reported", &reported),
            ("closed", &closed),
        ],
        origin_grain: Grain::Year,
        development_grain: Grain::Quarter,
        cumulative: true,
    })
    .unwrap();
    let e = medmal_case(CaseTrend::Given(0.1))
        .adjust(&tri)
        .unwrap_err()
        .to_string();
    assert!(e.contains("no cell at age 3"), "{e}");
    let e = SettlementRate {
        paid: "paid".into(),
        closed_counts: "closed".into(),
        ultimate_counts: UltimateCounts::Given(vec![50.0, 50.0]),
        disposal_rates: Selection::LatestDiagonal,
        interpolation: Interpolation::Exponential,
    }
    .adjust(&tri)
    .unwrap_err()
    .to_string();
    assert!(e.contains("no disposal rate at age 3"), "{e}");
}

/// No open claim on the latest diagonal at an age gives that age no average.
/// The older origins there keep their own case outstanding, and the
/// diagonal's cell its incurred, instead of falling to paid.
#[test]
fn no_open_claim_on_the_diagonal_keeps_the_case_outstanding_of_the_age() {
    let incurred: [&[f64]; 4] = [
        &[100., 150., 160., 165.],
        &[120., 170., 180.],
        &[125., 175.],
        &[130.],
    ];
    let paid: [&[f64]; 4] = [
        &[50., 100., 140., 150.],
        &[60., 110., 170.],
        &[65., 120.],
        &[70.],
    ];
    let open: [&[f64]; 4] = [&[5., 4., 0., 0.], &[6., 3., 2.], &[6., 0.], &[6.]];
    let tri = small(&[("incurred", &incurred), ("paid", &paid), ("open", &open)]);
    let case = CaseAdequacy {
        open_counts: OpenCounts::Column("open".into()),
        max_age: Some(24),
        ..medmal_case(CaseTrend::Given(0.1))
    };
    let got = cells(&case.adjust(&tri).unwrap(), "incurred");
    // 24 months: 2022 is on the diagonal with no open claim and 55 of case.
    assert_eq!(got[2][1], Some(175.0), "the diagonal keeps its incurred");
    assert_eq!(got[0][1], Some(150.0), "2020 keeps its own average");
    assert_eq!(got[1][1], Some(170.0), "2021 keeps its own average");
    // Past max_age: 2020 at 36 months has no open claim and 20 of case.
    assert_eq!(got[0][2], Some(160.0));
    // 12 months still restates to 2023's average, (130 - 70) / 6.
    close(
        got[0][0].unwrap(),
        50.0 + 5.0 * 10.0 / 1.331,
        1e-9,
        "2020 at 12",
    );
}

/// A selection that leaves out an observed age, or an origin with no usable
/// curve, must not lose an observed cell without saying so.
#[test]
fn the_settlement_adjustment_keeps_or_refuses_what_it_cannot_restate() {
    let paid: [&[f64]; 3] = [&[50., 100., 140.], &[0., 0.], &[70.]];
    let closed: [&[f64]; 3] = [&[5., 9., 11.], &[5., 10.], &[6.]];
    let tri = small(&[("paid", &paid), ("closed", &closed)]);
    let method = SettlementRate {
        paid: "paid".into(),
        closed_counts: "closed".into(),
        ultimate_counts: UltimateCounts::Given(vec![12.0, 12.0, 12.0]),
        disposal_rates: Selection::LatestDiagonal,
        interpolation: Interpolation::Exponential,
    };
    // 2021's paid is zero, so no exponential goes through it.
    let got = cells(&method.adjust(&tri).unwrap(), "paid");
    assert_eq!(got[1][0], Some(0.0));
    let e = SettlementRate {
        disposal_rates: Selection::Given(vec![(12, 0.5), (36, 0.9)]),
        ..method
    }
    .adjust(&tri)
    .unwrap_err()
    .to_string();
    assert!(e.contains("no disposal rate at age 24"), "{e}");
}

/// An origin the method has nothing for is an error, as it is for the chain
/// ladder's ultimate counts, and not a NaN in the totals.
#[test]
fn an_origin_without_the_columns_is_an_error_and_not_a_nan_total() {
    let nan = f64::NAN;
    let reported: [&[f64]; 2] = [&[100., 150.], &[120.]];
    let paid: [&[f64]; 2] = [&[50., 100.], &[nan]];
    let counts: [&[f64]; 2] = [&[5., 9.], &[nan]];
    let tri = small(&[
        ("reported", &reported),
        ("paid", &paid),
        ("counts", &counts),
    ]);
    let e = FrequencySeverity {
        amounts: "reported".into(),
        counts: "counts".into(),
        ultimate_counts: CountProjection::Counts(UltimateCounts::Given(vec![10.0, 10.0])),
        severity: SeverityProjection::Development(ChainLadder::default()),
    }
    .fit(&tri)
    .unwrap_err()
    .to_string();
    assert!(e.contains("2021"), "{e}");
    let disposal = Disposal {
        closed_counts: "counts".into(),
        paid: "paid".into(),
        ultimate_counts: UltimateCounts::Given(vec![10.0, 10.0]),
        disposal_rates: Selection::default(),
        severities: Selection::default(),
        tail_severity_from: None,
        trend: 0.0,
        base: None,
        factors: Vec::new(),
    };
    let e = disposal.fit(&tri).unwrap_err().to_string();
    assert!(e.contains("2021"), "{e}");
    let e = CaseOutstanding {
        paid: "paid".into(),
        incurred: "reported".into(),
        paid_on_case: Selection::default(),
        case_on_case: Selection::default(),
        paid_to_ultimate: 1.0,
    }
    .fit(&tri)
    .unwrap_err()
    .to_string();
    assert!(e.contains("2021"), "{e}");
}

/// A selected rate of 1 leaves no pattern to close an origin's open claims
/// on: an error, not a reserve of nothing.
#[test]
fn a_disposal_rate_of_one_with_claims_open_is_an_error() {
    let closed: [&[f64]; 2] = [&[8., 10.], &[8.]];
    let paid: [&[f64]; 2] = [&[50., 100.], &[60.]];
    let tri = small(&[("closed", &closed), ("paid", &paid)]);
    let e = Disposal {
        closed_counts: "closed".into(),
        paid: "paid".into(),
        ultimate_counts: UltimateCounts::Given(vec![10.0, 10.0]),
        disposal_rates: Selection::Given(vec![(12, 1.0), (24, 1.0)]),
        severities: Selection::default(),
        tail_severity_from: None,
        trend: 0.0,
        base: None,
        factors: Vec::new(),
    }
    .fit(&tri)
    .unwrap_err()
    .to_string();
    assert!(e.contains("2 claims to close"), "{e}");
}
