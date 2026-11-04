//! Recoveries, ALAE and ULAE against Friedland's printed exhibits
//! (`docs/design/reserving-v02.md`, decision 37).
//!
//! Friedland, *Estimating Unpaid Claims Using Basic Techniques* (CAS, version
//! 3, July 2010): chapter 14, Exhibit I (salvage and subrogation, Auto
//! Physical Damage Insurer), chapter 16, Exhibit I (ALAE, Auto Property
//! Damage Insurer) and chapter 17, Exhibits I to IV (ULAE, XYZ Insurer, New
//! Small Insurer and PQR Insurer). Issue #558 says chapters 14 and 15, from
//! an older table of contents: chapter 15 is the evaluation of techniques.
//!
//! The triangles in `data/friedland_auto_*.csv` are the printed cells, read
//! from the PDF by position and each checked against its printed diagonal.
//! The book's spreadsheets hold decimals behind them and multiply by factors
//! to ultimate rounded to three decimals, so the tests pass the printed
//! factors as selected patterns and allow for the printed rounding of the
//! claims: a unit on a printed cell and two on a printed total.
//!
//! The ratio method's cells are the ratios of the two columns, unrounded:
//! 5,957 / 16,822 x 1.009 x 16,893 is the printed 6,036, and the printed
//! 0.354 x 1.009 x 16,893 would be 6,034.

use std::collections::BTreeMap;

use prospicio_reserving::{
    Average, ChainLadder, Development, DevelopmentColumn, Grain, Lag, Long, Mack, Month,
    PatternKind, Period, RatioDevelopment, RatioFactors, SelectedPattern, Tail, Triangle, Ulae,
    UlaeInputs, UlaeWeights,
};

const AGES: [Lag; 11] = [12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 132];

/// The columns of a CSV in `data/` as one triangle, cumulative or not.
fn triangle(file: &str, cumulative: bool) -> Triangle {
    let path = format!("{}/data/{file}", env!("CARGO_MANIFEST_DIR"));
    let text = std::fs::read_to_string(path).unwrap();
    let mut lines = text.lines();
    let header: Vec<&str> = lines.next().unwrap().split(',').collect();
    let rows: Vec<Vec<f64>> = lines
        .map(|l| l.split(',').map(|v| v.parse().unwrap()).collect())
        .collect();
    let origin: Vec<Month> = rows.iter().map(|r| Month::january(r[0] as i32)).collect();
    let age: Vec<Lag> = rows.iter().map(|r| r[1] as Lag).collect();
    let columns: Vec<Vec<f64>> = (2..header.len())
        .map(|c| rows.iter().map(|r| r[c]).collect())
        .collect();
    let values: Vec<(&str, &[f64])> = (2..header.len())
        .map(|c| (header[c], columns[c - 2].as_slice()))
        .collect();
    Triangle::from_long(&Long {
        keys: &[],
        origin: &origin,
        development: DevelopmentColumn::Age(&age),
        values: &values,
        origin_grain: Grain::Year,
        development_grain: Grain::Year,
        cumulative,
    })
    .unwrap()
}

fn physical_damage() -> Triangle {
    triangle("friedland_auto_physical_damage.csv", true)
}

fn property_damage() -> Triangle {
    triangle("friedland_auto_property_damage.csv", true)
}

/// The book's printed factors to ultimate as a selected pattern.
fn printed_cdf(cdf: &[f64]) -> Development {
    Development {
        pattern: Some(SelectedPattern::new(
            PatternKind::AgeToUltimate,
            AGES.iter().copied().zip(cdf.iter().copied()).collect(),
        )),
        ..Default::default()
    }
}

/// `tri[column]` developed by the printed factors to ultimate.
fn ultimate(tri: &Triangle, column: &str, cdf: &[f64]) -> Vec<f64> {
    ChainLadder {
        development: printed_cdf(cdf),
        ..Default::default()
    }
    .fit(tri, column)
    .unwrap()
    .ultimate
}

fn close(what: &str, got: &[f64], want: &[f64], tolerance: f64) {
    assert_eq!(got.len(), want.len(), "{what}: lengths");
    for (k, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= tolerance,
            "{what}[{k}]: got {g}, want {w} within {tolerance}"
        );
    }
}

fn rounded(values: &[f64], digits: i32) -> Vec<f64> {
    let scale = 10f64.powi(digits);
    values.iter().map(|v| (v * scale).round() / scale).collect()
}

fn latest_5_volume() -> Development {
    Development {
        n_periods: Some(5),
        ..Default::default()
    }
}

// Chapter 14, Exhibit I: printed factors to ultimate, ages 12 to 132.
const SS_REPORTED_CDF: [f64; 11] = [
    1.067, 0.999, 1.001, 1.001, 1.001, 1.001, 1.0, 1.0, 1.0, 1.0, 1.0,
];
const SS_RECEIVED_CDF: [f64; 11] = [
    1.938, 1.022, 1.006, 1.005, 1.003, 1.002, 1.0, 1.0, 1.0, 1.0, 1.0,
];
const CLAIMS_REPORTED_CDF: [f64; 11] = [1.115, 1.001, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
const CLAIMS_PAID_CDF: [f64; 11] = [1.279, 1.005, 1.001, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
const SS_RATIO_CDF: [f64; 11] = [1.499, 1.009, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];

/// Sheet 6, column (9): the average of the two chain ladders.
fn ss_selected_claims() -> Vec<f64> {
    let tri = physical_damage();
    let reported = ultimate(&tri, "reported", &CLAIMS_REPORTED_CDF);
    let paid = ultimate(&tri, "paid", &CLAIMS_PAID_CDF);
    reported
        .iter()
        .zip(&paid)
        .map(|(r, p)| (r + p) / 2.0)
        .collect()
}

#[test]
fn salvage_and_subrogation_sheets_1_and_2_average_as_printed() {
    let tri = physical_damage();
    // The selected factors are the latest five years' volume-weighted
    // averages, and their product the printed factors to ultimate.
    let fit = ChainLadder {
        development: latest_5_volume(),
        ..Default::default()
    };
    let reported = fit.fit(&tri, "reported_ss").unwrap();
    close(
        "selected reported",
        &reported.ldf()[..7],
        &[1.068, 0.998, 1.000, 1.000, 1.000, 1.001, 1.000],
        0.001,
    );
    close("reported CDF", &reported.cdf, &SS_REPORTED_CDF, 0.0015);
    let received = fit.fit(&tri, "received_ss").unwrap();
    close(
        "selected received",
        &received.ldf()[..7],
        &[1.896, 1.016, 1.001, 1.002, 1.001, 1.002, 1.000],
        0.001,
    );
    close("received CDF", &received.cdf, &SS_RECEIVED_CDF, 0.0015);
}

#[test]
fn salvage_and_subrogation_sheet_3_projects_as_printed() {
    let tri = physical_damage();
    let reported = ultimate(&tri, "reported_ss", &SS_REPORTED_CDF);
    close(
        "reported",
        &reported,
        &[
            793.0, 1360.0, 2421.0, 3637.0, 4091.0, 4370.0, 5165.0, 5737.0, 5720.0, 6025.0, 5776.0,
        ],
        1.0,
    );
    let received = ultimate(&tri, "received_ss", &SS_RECEIVED_CDF);
    close(
        "received",
        &received,
        &[
            793.0, 1360.0, 2421.0, 3637.0, 4090.0, 4374.0, 5175.0, 5760.0, 5688.0, 6088.0, 5252.0,
        ],
        1.0,
    );
    close(
        "totals",
        &[reported.iter().sum(), received.iter().sum()],
        &[45_096.0, 44_639.0],
        2.0,
    );
}

#[test]
fn salvage_and_subrogation_sheet_6_selects_the_average() {
    let selected = ss_selected_claims();
    close(
        "selected claims",
        &selected,
        &[
            2864.0, 4697.0, 7902.0, 10319.0, 11137.0, 12527.0, 14536.0, 16837.0, 16952.0, 16893.0,
            16453.0,
        ],
        1.0,
    );
    close("total", &[selected.iter().sum()], &[131_117.0], 2.0);
}

#[test]
fn salvage_and_subrogation_sheets_7_and_8_develop_the_ratio() {
    let tri = physical_damage();
    let claims = ss_selected_claims();
    // Sheet 7's selected factors are the medial average of the latest five.
    let medial = RatioDevelopment {
        development: Development {
            average: Average::Simple,
            n_periods: Some(5),
            drop_high: 1,
            drop_low: 1,
            ..Default::default()
        },
        ..Default::default()
    }
    .fit(&tri, "received_ss", "paid", &claims)
    .unwrap();
    close(
        "medial average",
        &medial.averages[..6],
        &[1.486, 1.009, 1.000, 1.000, 1.000, 1.000],
        0.0005,
    );
    close("sheet 7 CDF", &medial.cdf, &SS_RATIO_CDF, 0.0015);
    close(
        "the first column of the ratio triangle",
        &rounded(
            &medial
                .ratios
                .iter()
                .map(|row| row[0].unwrap())
                .collect::<Vec<_>>(),
            3,
        ),
        &[
            0.157, 0.198, 0.166, 0.264, 0.257, 0.220, 0.235, 0.243, 0.233, 0.217, 0.210,
        ],
        0.0,
    );

    let fit = RatioDevelopment {
        development: printed_cdf(&SS_RATIO_CDF),
        selected: vec![(Period::year(2008), 0.345)],
        ..Default::default()
    }
    .fit(&tri, "received_ss", "paid", &claims)
    .unwrap();
    close(
        "latest ratio",
        &rounded(&fit.latest_ratio, 3),
        &[
            0.277, 0.290, 0.306, 0.352, 0.367, 0.348, 0.355, 0.340, 0.334, 0.354, 0.210,
        ],
        0.0,
    );
    close(
        "projected ratio",
        &rounded(&fit.projected_ratio, 3),
        &[
            0.277, 0.290, 0.306, 0.352, 0.367, 0.348, 0.355, 0.340, 0.334, 0.357, 0.315,
        ],
        0.0,
    );
    assert_eq!(fit.selected_ratio[10], 0.345);
    close(
        "projected S&S",
        &fit.ultimate,
        &[
            793.0, 1360.0, 2421.0, 3637.0, 4090.0, 4365.0, 5165.0, 5731.0, 5658.0, 6036.0, 5676.0,
        ],
        1.0,
    );
    close("total", &[fit.total_ultimate()], &[44_934.0], 2.0);
}

#[test]
fn salvage_and_subrogation_sheet_9_gives_the_recoverables() {
    let tri = physical_damage();
    let received: f64 = 41_879.0;
    let reported = ultimate(&tri, "reported_ss", &SS_REPORTED_CDF);
    let developed = ultimate(&tri, "received_ss", &SS_RECEIVED_CDF);
    let ratio = RatioDevelopment {
        development: printed_cdf(&SS_RATIO_CDF),
        selected: vec![(Period::year(2008), 0.345)],
        ..Default::default()
    }
    .fit(&tri, "received_ss", "paid", &ss_selected_claims())
    .unwrap();
    assert_eq!(ratio.latest.iter().sum::<f64>(), received);
    close(
        "recoverable",
        &ratio.unpaid(),
        &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6.0, 0.0, 3.0, 79.0, 2966.0],
        1.0,
    );
    close(
        "total recoverables",
        &[
            reported.iter().sum::<f64>() - received,
            developed.iter().sum::<f64>() - received,
            ratio.total_unpaid(),
        ],
        &[3_216.0, 2_760.0, 3_054.0],
        2.0,
    );
}

// Chapter 16, Exhibit I: printed factors to ultimate, ages 12 to 132.
const ALAE_REPORTED_CDF: [f64; 11] = [
    1.367, 1.169, 1.106, 1.066, 1.045, 1.008, 1.002, 1.0, 1.0, 1.0, 1.0,
];
const ALAE_PAID_CDF: [f64; 11] = [
    2.138, 1.241, 1.155, 1.096, 1.058, 1.028, 1.013, 1.009, 1.007, 1.005, 1.005,
];
const ONLY_REPORTED_CDF: [f64; 11] = [
    1.101, 0.990, 0.989, 0.991, 0.993, 0.998, 0.999, 0.999, 1.0, 1.0, 1.0,
];
const ONLY_PAID_CDF: [f64; 11] = [
    1.584, 1.029, 1.007, 1.004, 1.002, 1.001, 1.001, 1.001, 1.0, 1.0, 1.0,
];
const ALAE_RATIO_CDF: [f64; 11] = [
    1.332, 1.201, 1.140, 1.086, 1.050, 1.021, 1.007, 1.003, 1.002, 1.0, 1.0,
];

/// Sheet 6, column (9).
fn alae_selected_claims() -> Vec<f64> {
    let tri = property_damage();
    let reported = ultimate(&tri, "reported", &ONLY_REPORTED_CDF);
    let paid = ultimate(&tri, "paid", &ONLY_PAID_CDF);
    reported
        .iter()
        .zip(&paid)
        .map(|(r, p)| (r + p) / 2.0)
        .collect()
}

fn latest_3_simple() -> Development {
    Development {
        average: Average::Simple,
        n_periods: Some(3),
        ..Default::default()
    }
}

fn alae_selected_ratios() -> Vec<(Period, f64)> {
    vec![(Period::year(2007), 0.0077), (Period::year(2008), 0.0077)]
}

#[test]
fn alae_sheets_1_to_3_develop_the_dollars() {
    let tri = property_damage();
    let latest_3 = Development {
        n_periods: Some(3),
        ..Default::default()
    };
    let reported = ChainLadder {
        development: latest_3.clone(),
        ..Default::default()
    }
    .fit(&tri, "reported_alae")
    .unwrap();
    close("reported CDF", &reported.cdf, &ALAE_REPORTED_CDF, 0.0015);
    // The paid tail of 1.005 from the reported to paid ratios.
    let paid = ChainLadder {
        development: latest_3,
        tail: Tail::from(1.005),
    }
    .fit(&tri, "paid_alae")
    .unwrap();
    close("paid CDF", &paid.cdf, &ALAE_PAID_CDF, 0.0015);
    let reported = ultimate(&tri, "reported_alae", &ALAE_REPORTED_CDF);
    close(
        "reported",
        &reported,
        &[
            1092.0, 1051.0, 862.0, 1007.0, 1112.0, 1152.0, 1078.0, 1100.0, 1325.0, 1866.0, 2127.0,
        ],
        1.0,
    );
    let paid = ultimate(&tri, "paid_alae", &ALAE_PAID_CDF);
    close(
        "paid",
        &paid,
        &[
            1089.0, 1050.0, 867.0, 1007.0, 1112.0, 1115.0, 1046.0, 1082.0, 1307.0, 1804.0, 2035.0,
        ],
        1.0,
    );
    // Sheet 11, columns (8) and (9).
    close(
        "total unpaid",
        &[
            reported.iter().sum::<f64>() - 11_685.0,
            paid.iter().sum::<f64>() - 11_685.0,
        ],
        &[2_088.0, 1_832.0],
        2.0,
    );
}

#[test]
fn alae_sheet_6_selects_the_average_with_factors_below_1() {
    let tri = property_damage();
    let reported = ultimate(&tri, "reported", &ONLY_REPORTED_CDF);
    assert!(reported[3] < 137_056.0, "the factor below 1 lowers 2001");
    close(
        "selected claims only",
        &alae_selected_claims(),
        &[
            108_731.0, 116_079.0, 116_639.0, 137_026.0, 141_544.0, 163_301.0, 152_796.0, 153_181.0,
            177_121.0, 176_560.0, 195_485.0,
        ],
        1.0,
    );
}

#[test]
fn alae_sheets_7_and_8_develop_the_ratio() {
    let tri = property_damage();
    let claims = alae_selected_claims();
    // Sheet 7: the simple average of the latest three, and no tail.
    let averaged = RatioDevelopment {
        development: latest_3_simple(),
        ..Default::default()
    }
    .fit(&tri, "paid_alae", "paid", &claims)
    .unwrap();
    close(
        "selected",
        &averaged.factors,
        &[
            1.109, 1.053, 1.050, 1.034, 1.029, 1.013, 1.004, 1.001, 1.002, 1.000,
        ],
        0.0005,
    );
    close("sheet 7 CDF", &averaged.cdf, &ALAE_RATIO_CDF, 0.0015);

    let fit = RatioDevelopment {
        development: printed_cdf(&ALAE_RATIO_CDF),
        selected: alae_selected_ratios(),
        ..Default::default()
    }
    .fit(&tri, "paid_alae", "paid", &claims)
    .unwrap();
    close(
        "projected ratio",
        &rounded(&fit.projected_ratio, 4),
        &[
            0.0100, 0.0090, 0.0074, 0.0073, 0.0078, 0.0068, 0.0068, 0.0070, 0.0073, 0.0102, 0.0102,
        ],
        0.0,
    );
    close(
        "projected ALAE",
        &fit.ultimate,
        &[
            1084.0, 1045.0, 861.0, 1001.0, 1106.0, 1108.0, 1040.0, 1072.0, 1302.0, 1360.0, 1505.0,
        ],
        1.0,
    );
    close("total", &[fit.total_ultimate()], &[12_485.0], 2.0);
    // Sheet 11, column (10): the selected 0.0077 puts 2007 below its paid.
    close(
        "unpaid",
        &fit.unpaid(),
        &[
            0.0, 0.0, 0.0, 3.0, 8.0, 23.0, 51.0, 85.0, 170.0, -94.0, 553.0,
        ],
        1.0,
    );
    close("total unpaid", &[fit.total_unpaid()], &[800.0], 2.0);
}

#[test]
fn alae_sheets_9_and_10_add_the_differences() {
    let tri = property_damage();
    let claims = alae_selected_claims();
    let fit = RatioDevelopment {
        development: latest_3_simple(),
        factors: RatioFactors::Additive {
            selected: vec![],
            tail: 0.0,
        },
        ..Default::default()
    }
    .fit(&tri, "paid_alae", "paid", &claims)
    .unwrap();
    close(
        "selected",
        &rounded(&fit.factors, 4),
        &[
            0.0007, 0.0003, 0.0003, 0.0002, 0.0002, 0.0001, 0.0000, 0.0000, 0.0000, 0.0000,
        ],
        0.0,
    );
    close(
        "additive CDF",
        &rounded(&fit.cdf, 4),
        &[
            0.0019, 0.0012, 0.0009, 0.0006, 0.0004, 0.0002, 0.0001, 0.0000, 0.0000, 0.0000, 0.0000,
        ],
        0.0,
    );
    close(
        "projected ratio",
        &rounded(&fit.projected_ratio, 4),
        &[
            0.0100, 0.0090, 0.0074, 0.0073, 0.0078, 0.0068, 0.0068, 0.0070, 0.0073, 0.0097, 0.0096,
        ],
        0.0,
    );
    close(
        "projected ALAE",
        &fit.ultimate,
        &[
            1084.0, 1045.0, 862.0, 1003.0, 1108.0, 1112.0, 1046.0, 1077.0, 1300.0, 1709.0, 1870.0,
        ],
        1.0,
    );
    close("total", &[fit.total_ultimate()], &[13_215.0], 2.0);
    // Sheet 11, column (11).
    close(
        "unpaid",
        &fit.unpaid(),
        &[
            0.0, 0.0, 1.0, 5.0, 10.0, 27.0, 57.0, 90.0, 168.0, 255.0, 918.0,
        ],
        1.0,
    );
    close("total unpaid", &[fit.total_unpaid()], &[1_530.0], 2.0);
}

#[test]
fn a_recovery_triangle_runs_through_as_negative_increments() {
    // Reported S&S falls as case estimates of recoveries are taken down.
    let cumulative = physical_damage();
    let incremental = cumulative.to_incremental();
    let mut negative = 0;
    for o in 0..11 {
        for d in 0..11 {
            if incremental.get(0, 0, o, d).is_some_and(|v| v < 0.0) {
                negative += 1;
            }
        }
    }
    assert!(negative > 10, "{negative} negative increments");

    let wide = incremental.to_long();
    let columns: Vec<(&str, &[f64])> = wide
        .values
        .iter()
        .map(|(name, v)| (name.as_str(), v.as_slice()))
        .collect();
    let rebuilt = Triangle::from_long(&Long {
        keys: &[],
        origin: &wide.origin,
        development: DevelopmentColumn::Age(&wide.development),
        values: &columns,
        origin_grain: Grain::Year,
        development_grain: Grain::Year,
        cumulative: false,
    })
    .unwrap();

    let cl = ChainLadder::default();
    let from_increments = cl.fit(&rebuilt, "reported_ss").unwrap();
    let from_cumulative = cl.fit(&cumulative, "reported_ss").unwrap();
    assert_eq!(from_increments.ultimate, from_cumulative.ultimate);
    assert!(from_increments.ldf()[1] < 1.0, "a link ratio below 1");

    // The printed factor to ultimate of 0.999 at 24 months lowers 2007.
    let printed = ChainLadder {
        development: printed_cdf(&SS_REPORTED_CDF),
        ..Default::default()
    }
    .fit(&rebuilt, "reported_ss")
    .unwrap();
    assert!(printed.cdf[1] < 1.0, "a factor to ultimate below 1");
    assert!(printed.reserves()[9] < 0.0, "a negative reserve");

    // Mack's model takes them too, and its reserve is the chain ladder's.
    let mack = Mack::default().fit(&rebuilt, "reported_ss").unwrap();
    assert!((mack.chain_ladder.total_reserve() - from_cumulative.total_reserve()).abs() < 1e-9);
    assert!(mack.total_standard_error.is_finite() && mack.total_standard_error > 0.0);

    // The ratio of the two takes the incremental triangle as it does the
    // cumulative one: the ratio is of the cumulative values.
    let claims = vec![1.0; 11];
    let ratio = RatioDevelopment::default();
    assert_eq!(
        ratio.fit(&rebuilt, "received_ss", "paid", &claims).unwrap(),
        ratio
            .fit(&cumulative, "received_ss", "paid", &claims)
            .unwrap(),
    );
}

#[test]
fn a_cumulative_recovery_below_zero_still_develops() {
    // Salvage paid back, so a cumulative value below zero: the volume and
    // simple averages are defined and the geometric one is not.
    let origin = [2020, 2020, 2020, 2021, 2021, 2022].map(Month::january);
    let tri = Triangle::from_long(&Long {
        keys: &[],
        origin: &origin,
        development: DevelopmentColumn::Age(&[12, 24, 36, 12, 24, 12]),
        values: &[("recoveries", &[-10.0, 20.0, 30.0, 5.0, 10.0, 6.0])],
        origin_grain: Grain::Year,
        development_grain: Grain::Year,
        cumulative: true,
    })
    .unwrap();
    let cl = ChainLadder::default().fit(&tri, "recoveries").unwrap();
    close("volume", &cl.ldf()[..1], &[30.0 / -5.0], 1e-12);
    let geometric = ChainLadder {
        development: Development {
            average: Average::Geometric,
            ..Default::default()
        },
        ..Default::default()
    };
    assert!(geometric.fit(&tri, "recoveries").is_err());
    // Mack's variance needs positive values: on these it cannot be estimated.
    assert!(Mack::default().fit(&tri, "recoveries").is_err());
}

// Chapter 17, Exhibits I and II: XYZ Insurer, 2004 to 2008.
const XYZ_ULAE: [f64; 5] = [14.352e6, 15.321e6, 16.870e6, 17.112e6, 17.331e6];
const XYZ_PAID: [f64; 5] = [333e6, 358e6, 334e6, 347e6, 391e6];
const XYZ_INCURRED: [f64; 5] = [535.213e6, 492.265e6, 435.985e6, 432.966e6, 475.300e6];

fn xyz<'a>() -> UlaeInputs<'a> {
    UlaeInputs {
        paid_ulae: &XYZ_ULAE,
        paid: &XYZ_PAID,
        reported: &XYZ_INCURRED,
        case_outstanding: &[603e6],
        ibnr: &[316e6],
        ..Default::default()
    }
}

#[test]
fn ulae_exhibit_i_is_the_classical_technique() {
    let model = Ulae::classical(Some(0.045));
    let total = model.estimate(&xyz()).unwrap();
    close(
        "ratios",
        &rounded(&total.ratios, 3),
        &[0.043, 0.043, 0.051, 0.049, 0.044],
        0.0,
    );
    close(
        "total ratio",
        &rounded(&[total.total_ratio], 3),
        &[0.046],
        0.0,
    );
    close("total IBNR", &[total.total()], &[27_787_500.0], 1e-6);
    // Pure IBNR of 5% of accident year 2008's ultimate claims of 380 million.
    let pure = [0.05 * 380e6];
    let pure = model
        .estimate(&UlaeInputs {
            pure_ibnr: Some(&pure),
            ..xyz()
        })
        .unwrap();
    close("pure IBNR", &[pure.total()], &[21_105_000.0], 1e-6);
}

#[test]
fn ulae_exhibit_ii_is_kittels_refinement() {
    let model = Ulae::kittel(Some(0.040));
    let total = model.estimate(&xyz()).unwrap();
    close(
        "average of paid and incurred",
        &total.basis,
        &[
            434_106_500.0,
            425_132_500.0,
            384_992_500.0,
            389_983_000.0,
            433_150_000.0,
        ],
        1e-6,
    );
    close(
        "ratios",
        &rounded(&total.ratios, 3),
        &[0.033, 0.036, 0.044, 0.044, 0.040],
        0.0,
    );
    close(
        "total ratio",
        &rounded(&[total.total_ratio], 3),
        &[0.039],
        0.0,
    );
    close("total IBNR", &[total.total()], &[24_700_000.0], 1e-6);
    let pure = [19e6];
    let pure = model
        .estimate(&UlaeInputs {
            pure_ibnr: Some(&pure),
            ..xyz()
        })
        .unwrap();
    close("pure IBNR", &[pure.total()], &[18_760_000.0], 1e-6);
}

#[test]
fn ulae_exhibit_iii_is_mango_and_allen_on_expected_paid_claims() {
    // Sheet 1: the premium's expected claims times the benchmark pattern.
    let expected_claims = [
        4_300_000.0 * 0.55,
        4_250_000.0 * 0.55,
        4_420_000.0 * 0.55,
        3_985_000.0 * 0.55,
    ];
    let pattern = [0.12, 0.15, 0.15, 0.15];
    let expected_paid: Vec<f64> = (0..4)
        .map(|year| {
            (0..=year)
                .map(|origin| expected_claims[origin] * pattern[year - origin])
                .sum()
        })
        .collect();
    close(
        "expected paid",
        &expected_paid,
        &[283_800.0, 635_250.0, 997_095.0, 1_333_035.0],
        1e-6,
    );
    let inputs = UlaeInputs {
        paid_ulae: &[55_000.0, 62_500.0, 70_000.0, 80_000.0],
        paid: &expected_paid,
        case_outstanding: &[225_000.0],
        ibnr: &[6_430_000.0],
        ..Default::default()
    };
    let model = Ulae::classical(Some(0.07));
    let total = model.estimate(&inputs).unwrap();
    close(
        "ratios",
        &rounded(&total.ratios, 3),
        &[0.194, 0.098, 0.070, 0.060],
        0.0,
    );
    close(
        "total ratio",
        &rounded(&[total.total_ratio], 3),
        &[0.082],
        0.0,
    );
    close("total IBNR", &[total.total()], &[457_975.0], 1e-6);
    // Pure IBNR of 5% of accident year 2008's expected claims.
    let pure = [0.05 * expected_claims[3]];
    let pure = model
        .estimate(&UlaeInputs {
            pure_ibnr: Some(&pure),
            ..inputs
        })
        .unwrap();
    close("pure IBNR", &[pure.total()], &[236_761.0], 0.5);
}

// Chapter 17, Exhibit IV: PQR Insurer, 2003 to 2008, in thousands.
const PQR_ULAE: [f64; 6] = [1978.0, 4820.0, 8558.0, 12039.0, 13143.0, 15286.0];
const PQR_PAID: [f64; 6] = [4590.0, 14600.0, 38390.0, 58297.0, 86074.0, 105466.0];
const PQR_REPORTED: [f64; 6] = [19534.0, 57125.0, 85521.0, 128672.0, 145070.0, 163626.0];
const PQR_ULT_REPORTED: [f64; 6] = [27200.0, 76700.0, 106900.0, 154300.0, 163100.0, 176400.0];
const PQR_ULTIMATE: [f64; 6] = [28600.0, 79200.0, 108400.0, 156700.0, 163400.0, 177100.0];
const PQR_IBNR_BY_YEAR: [f64; 6] = [257.0, 1742.0, 5095.0, 16140.0, 34477.0, 56141.0];

#[test]
fn ulae_exhibit_iv_sheet_2_compares_the_classical_and_kittel() {
    let inputs = UlaeInputs {
        paid_ulae: &PQR_ULAE,
        paid: &PQR_PAID,
        reported: &PQR_REPORTED,
        case_outstanding: &[292_130.0],
        ibnr: &[113_853.0],
        ..Default::default()
    };
    let classical = Ulae::classical(Some(0.160)).estimate(&inputs).unwrap();
    close(
        "traditional",
        &rounded(&classical.ratios, 3),
        &[0.431, 0.330, 0.223, 0.207, 0.153, 0.145],
        0.0,
    );
    close(
        "classical total",
        &rounded(&[classical.total_ratio], 3),
        &[0.182],
        0.0,
    );
    close("classical", &[classical.total()], &[41_587.0], 0.5);
    let kittel = Ulae::kittel(Some(0.115)).estimate(&inputs).unwrap();
    close(
        "Kittel",
        &rounded(&kittel.ratios, 3),
        &[0.164, 0.134, 0.138, 0.129, 0.114, 0.114],
        0.0,
    );
    close(
        "Kittel total",
        &rounded(&[kittel.total_ratio], 3),
        &[0.123],
        0.0,
    );
    close("Kittel", &[kittel.total()], &[29_891.0], 0.5);
}

/// The Conger and Nolibos inputs of sheets 3 and 4 by the printed `weights`.
fn pqr_generalized(opening: f64) -> prospicio_reserving::UlaeEstimate {
    // Exhibit IV, Sheet 1 in total: L - R is 8,800, and L - P is the case
    // outstanding and IBNR of sheet 2 to the rounding of the IBNR by 1.
    let pure = [713_400.0 - 704_600.0];
    Ulae::conger_nolibos(UlaeWeights::new(opening, 1.0 - opening), Some(0.100))
        .estimate(&UlaeInputs {
            paid_ulae: &PQR_ULAE,
            paid: &PQR_PAID,
            reported: &PQR_ULT_REPORTED,
            ultimate: &PQR_ULTIMATE,
            case_outstanding: &[292_130.0],
            ibnr: &[113_853.0],
            pure_ibnr: Some(&pure),
            ..Default::default()
        })
        .unwrap()
}

#[test]
fn ulae_exhibit_iv_sheet_3_is_the_generalized_approach_at_60_40() {
    let fit = pqr_generalized(0.6);
    close(
        "basis",
        &fit.basis,
        &[
            18_156.0, 51_860.0, 79_496.0, 115_899.0, 132_290.0, 148_026.0,
        ],
        0.5,
    );
    close(
        "ratios",
        &rounded(&fit.ratios, 3),
        &[0.109, 0.093, 0.108, 0.104, 0.099, 0.103],
        0.0,
    );
    close(
        "total ratio",
        &rounded(&[fit.total_ratio], 3),
        &[0.102],
        0.0,
    );
    close("total basis", &[fit.basis.iter().sum()], &[545_727.0], 0.5);
    close(
        "expected",
        &[fit.expected_claims.unwrap()],
        &[15_516.0],
        0.5,
    );
    close("BF", &[fit.bornhuetter_ferguson.unwrap()], &[16_767.0], 0.5);
    close("development", &[fit.development.unwrap()], &[17_152.0], 0.5);
    // By liabilities, `W* (U1 (L - R) + U2 (L - P))`, the same number.
    close("unpaid", &[fit.total()], &[16_767.0], 0.5);
}

#[test]
fn ulae_exhibit_iv_sheet_4_is_the_generalized_approach_at_70_30() {
    let fit = pqr_generalized(0.7);
    close(
        "basis",
        &fit.basis,
        &[
            20_417.0, 58_070.0, 86_347.0, 125_499.0, 139_992.0, 155_120.0,
        ],
        0.5,
    );
    close(
        "ratios",
        &rounded(&fit.ratios, 3),
        &[0.097, 0.083, 0.099, 0.096, 0.094, 0.099],
        0.0,
    );
    close(
        "total ratio",
        &rounded(&[fit.total_ratio], 3),
        &[0.095],
        0.0,
    );
    close("total basis", &[fit.basis.iter().sum()], &[585_445.0], 0.5);
    close(
        "expected",
        &[fit.expected_claims.unwrap()],
        &[15_516.0],
        0.5,
    );
    close("BF", &[fit.bornhuetter_ferguson.unwrap()], &[12_795.0], 0.5);
    close("development", &[fit.development.unwrap()], &[12_201.0], 0.5);
    close("unpaid", &[fit.total()], &[12_795.0], 0.5);
}

#[test]
fn ulae_exhibit_iv_sheet_5_is_the_simplification() {
    // The accident year's ultimate stands in for R. The case outstanding
    // and IBNR of sheet 2 are L - P, 405,983.
    let model = Ulae::conger_nolibos(UlaeWeights::new(0.6, 0.4), Some(0.100));
    for (share, pure_printed, unpaid_printed) in
        [(0.04, 7_084.0, 16_664.0), (0.06, 10_626.0, 16_877.0)]
    {
        let pure = [share * 177_100.0];
        close("pure IBNR", &pure, &[pure_printed], 0.5);
        let fit = model
            .estimate(&UlaeInputs {
                paid_ulae: &PQR_ULAE,
                paid: &PQR_PAID,
                reported: &PQR_ULTIMATE,
                case_outstanding: &[292_130.0],
                ibnr: &[113_853.0],
                pure_ibnr: Some(&pure),
                ..Default::default()
            })
            .unwrap();
        close(
            "basis",
            &fit.basis,
            &[
                18_996.0, 53_360.0, 80_396.0, 117_339.0, 132_470.0, 148_446.0,
            ],
            0.5,
        );
        close(
            "ratios",
            &rounded(&fit.ratios, 3),
            &[0.104, 0.090, 0.106, 0.103, 0.099, 0.103],
            0.0,
        );
        close(
            "total ratio",
            &rounded(&[fit.total_ratio], 3),
            &[0.101],
            0.0,
        );
        close("total basis", &[fit.basis.iter().sum()], &[551_007.0], 0.5);
        close("unpaid", &[fit.total()], &[unpaid_printed], 0.5);
    }
}

#[test]
fn ulae_by_accident_year_sums_to_the_total() {
    // Friedland applies the ratio to totals; every form is linear, so rows
    // by accident year give the same total. PQR's IBNR by accident year
    // (Exhibit IV, Sheet 1) with the case outstanding spread over them.
    let total_ibnr: f64 = PQR_IBNR_BY_YEAR.iter().sum();
    let case: Vec<f64> = PQR_IBNR_BY_YEAR
        .iter()
        .map(|i| 292_130.0 * i / total_ibnr)
        .collect();
    let by_year = UlaeInputs {
        paid_ulae: &PQR_ULAE,
        paid: &PQR_PAID,
        case_outstanding: &case,
        ibnr: &PQR_IBNR_BY_YEAR,
        ..Default::default()
    };
    let fit = Ulae::classical(Some(0.160)).estimate(&by_year).unwrap();
    assert_eq!(fit.unpaid.len(), 6);
    let whole = Ulae::classical(Some(0.160))
        .estimate(&UlaeInputs {
            case_outstanding: &[292_130.0],
            ibnr: &[total_ibnr],
            ..by_year
        })
        .unwrap();
    close(
        "the sum of the years",
        &[fit.total()],
        &[whole.total()],
        1e-6,
    );
}

#[test]
fn the_data_are_the_printed_diagonals() {
    let tri = physical_damage();
    let totals: BTreeMap<&str, f64> = BTreeMap::from([
        ("reported_ss", 44_718.0),
        ("received_ss", 41_879.0),
        ("reported", 129_370.0),
        ("paid", 127_456.0),
    ]);
    for (column, total) in totals {
        let got: f64 = ChainLadder::default()
            .fit(&tri, column)
            .unwrap()
            .latest
            .iter()
            .sum();
        assert!((got - total).abs() <= 1.0, "{column}: {got}");
    }
    let tri = property_damage();
    let totals: BTreeMap<&str, f64> = BTreeMap::from([
        ("reported_alae", 12_679.0),
        ("paid_alae", 11_685.0),
        ("reported", 1_625_091.0),
        ("paid", 1_560_626.0),
    ]);
    for (column, total) in totals {
        let got: f64 = ChainLadder::default()
            .fit(&tri, column)
            .unwrap()
            .latest
            .iter()
            .sum();
        assert!((got - total).abs() <= 1.0, "{column}: {got}");
    }
}
